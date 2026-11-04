// C library imports. The Switch libc is Nintendo's musl port; AAPCS64 is identical on both sides,
// and the structs these functions use (struct tm, FILE*, va_list, wchar_t = 4 bytes) are layout
// compatible with bionic, so most functions alias directly to the host libc. Exceptions:
//  * functions that allocate memory the game will free (strdup, memalign) use the guest allocator,
//  * functions that take guest callbacks (qsort, bsearch, atexit) call them through GuestCall,
//  * setjmp/longjmp use a musl-layout implementation (arch/arm64/abi.S),
//  * stdout printing is redirected to the log.

#include <cctype>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <mutex>
#include <vector>

#include "arch/arch.h"
#include "common/log.h"
#include "hle/registry.h"
#include "runtime/runtime.h"

extern "C" {
// compiler-rt / libgcc soft-float quad helpers (long double is IEEE binary128 on AArch64).
void __addtf3() __attribute__((weak));
void __floatditf() __attribute__((weak));
void __lttf2() __attribute__((weak));
void __multf3() __attribute__((weak));
int __cxa_guard_acquire(uint64_t*);
void __cxa_guard_release(uint64_t*);
#if defined(__BIONIC__)
int* __errno();
#endif
}

namespace {
constexpr const char* kTag = "libc";

struct LibcEntry {
    const char* name;
    void* fn;
};

#define C(name, type) {#name, reinterpret_cast<void*>(static_cast<type>(&::name))}
#define P(name) {#name, reinterpret_cast<void*>(&::name)}

using d_d = double (*)(double);
using f_f = float (*)(float);
using d_dd = double (*)(double, double);
using f_ff = float (*)(float, float);
using i_i = int (*)(int);

const LibcEntry kDirect[] = {
    // memory / strings
    C(memcpy, void* (*)(void*, const void*, size_t)), C(memmove, void* (*)(void*, const void*, size_t)),
    C(memset, void* (*)(void*, int, size_t)), C(memcmp, int (*)(const void*, const void*, size_t)),
    C(memchr, const void* (*)(const void*, int, size_t)), C(strlen, size_t(*)(const char*)),
    C(strnlen, size_t(*)(const char*, size_t)), C(strcmp, int (*)(const char*, const char*)),
    C(strncmp, int (*)(const char*, const char*, size_t)), C(strcpy, char* (*)(char*, const char*)),
    C(strncpy, char* (*)(char*, const char*, size_t)), C(strcat, char* (*)(char*, const char*)),
    C(strncat, char* (*)(char*, const char*, size_t)), C(strchr, const char* (*)(const char*, int)),
    C(strrchr, const char* (*)(const char*, int)), C(strstr, const char* (*)(const char*, const char*)),
    C(strpbrk, const char* (*)(const char*, const char*)), C(strcspn, size_t(*)(const char*, const char*)),
    C(strcoll, int (*)(const char*, const char*)), C(strcasecmp, int (*)(const char*, const char*)),
    C(strncasecmp, int (*)(const char*, const char*, size_t)), C(strerror, char* (*)(int)),
    C(wcslen, size_t(*)(const wchar_t*)), C(wcscmp, int (*)(const wchar_t*, const wchar_t*)),
    C(wcsncmp, int (*)(const wchar_t*, const wchar_t*, size_t)),
    C(wcsspn, size_t(*)(const wchar_t*, const wchar_t*)), C(wmemset, wchar_t* (*)(wchar_t*, wchar_t, size_t)),
    // conversions
    C(atoi, int (*)(const char*)), C(atol, long (*)(const char*)), C(atoll, long long (*)(const char*)),
    C(atof, double (*)(const char*)), C(strtol, long (*)(const char*, char**, int)),
    C(strtoul, unsigned long (*)(const char*, char**, int)), C(strtoll, long long (*)(const char*, char**, int)),
    C(strtoull, unsigned long long (*)(const char*, char**, int)), C(strtod, double (*)(const char*, char**)),
    C(abs, i_i),
    // ctype
    C(tolower, i_i), C(toupper, i_i), C(isupper, i_i), C(islower, i_i), C(isspace, i_i), C(isalnum, i_i),
    C(isalpha, i_i), C(iscntrl, i_i), C(ispunct, i_i), C(isxdigit, i_i),
    // formatted I/O (variadic: identical AAPCS64 convention on both sides)
    P(sprintf), P(snprintf), P(vsnprintf), P(vsprintf), P(sscanf), P(swprintf), P(fscanf),
    // stdio on host FILE* handles (opaque to the guest)
    P(fopen), P(fclose), P(fflush), P(fread), P(fwrite), P(fseek), P(ftell), P(fseeko), P(ftello), P(fgets),
    P(fgetws), P(getc), P(ungetc), P(feof), P(ferror), P(fputc), P(remove), P(rename),
    // time
    P(time), P(clock), P(difftime), P(mktime), P(gmtime), P(localtime), P(gmtime_r), P(localtime_r), P(timegm),
    P(strftime),
    // misc
    P(getenv), P(setlocale), P(rand), P(srand), P(lrand48), P(srand48),
    // math
    C(sqrt, d_d), C(sqrtf, f_f), C(pow, d_dd), C(powf, f_ff), C(sin, d_d), C(cos, d_d), C(tan, d_d),
    C(sinf, f_f), C(cosf, f_f), C(tanf, f_f), C(asin, d_d), C(acos, d_d), C(atan, d_d), C(asinf, f_f),
    C(acosf, f_f), C(atanf, f_f), C(atan2, d_dd), C(atan2f, f_ff), C(sinhf, f_f), C(coshf, f_f), C(tanhf, f_f),
    C(exp, d_d), C(expf, f_f), C(exp2f, f_f), C(log, d_d), C(logf, f_f), C(log10, d_d), C(log10f, f_f),
    C(fmod, d_dd), C(fmodf, f_ff), C(frexp, double (*)(double, int*)), C(frexpf, float (*)(float, int*)),
    C(ldexp, double (*)(double, int)), C(ldexpf, float (*)(float, int)), C(modf, double (*)(double, double*)),
    C(modff, float (*)(float, float*)),
    // C++ ABI runtime
    P(__cxa_guard_acquire), P(__cxa_guard_release),
};

#undef C
#undef P

struct Registrar {
    Registrar() {
        for (const auto& e : kDirect) dsr::hle::Register(e.name, e.fn, false);
        const LibcEntry quad[] = {{"__addtf3", reinterpret_cast<void*>(&__addtf3)},
                                  {"__floatditf", reinterpret_cast<void*>(&__floatditf)},
                                  {"__lttf2", reinterpret_cast<void*>(&__lttf2)},
                                  {"__multf3", reinterpret_cast<void*>(&__multf3)}};
        for (const auto& e : quad)
            if (e.fn) dsr::hle::Register(e.name, e.fn, false);
    }
} g_registrar;

// ---- stdio globals: musl's `FILE *const stdout` etc. are pointer-sized data objects ----------------
FILE* g_stdin = stdin;
FILE* g_stdout = stdout;
FILE* g_stderr = stderr;

bool IsConsole(FILE* f) { return f == g_stdout || f == g_stderr; }

void LogFormatted(const char* fmt, va_list ap) {
    char buf[2048];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    if (n) LOGI("guest", "%s", buf);
}

// ---- atexit ---------------------------------------------------------------------------------------
std::mutex g_atexit_mutex;
std::vector<const void*> g_atexit;
}  // namespace

HLE_DATA(stdin, g_stdin);
HLE_DATA(stdout, g_stdout);
HLE_DATA(stderr, g_stderr);

#if defined(__BIONIC__)
HLE_FUNC(__errno_location, int*, libc_errno_location, ()) { return __errno(); }
#else
HLE_FUNC(__errno_location, int*, libc_errno_location, ()) { return &errno; }
#endif

HLE_FUNC(printf, int, libc_printf, (const char* fmt, ...)) {
    va_list ap;
    va_start(ap, fmt);
    LogFormatted(fmt, ap);
    va_end(ap);
    return 0;
}

HLE_FUNC(fprintf, int, libc_fprintf, (FILE * f, const char* fmt, ...)) {
    va_list ap;
    va_start(ap, fmt);
    int r = 0;
    if (IsConsole(f))
        LogFormatted(fmt, ap);
    else
        r = vfprintf(f, fmt, ap);
    va_end(ap);
    return r;
}

HLE_FUNC(puts, int, libc_puts, (const char* s)) {
    LOGI("guest", "%s", s);
    return 1;
}

HLE_FUNC(fputs, int, libc_fputs, (const char* s, FILE* f)) {
    if (IsConsole(f)) {
        LOGI("guest", "%s", s);
        return 1;
    }
    return fputs(s, f);
}

HLE_FUNC(strdup, char*, libc_strdup, (const char* s)) {
    size_t n = strlen(s) + 1;
    auto* p = static_cast<char*>(dsr::GuestMalloc(n));
    if (p) memcpy(p, s, n);
    return p;
}

HLE_FUNC(memalign, void*, libc_memalign, (size_t align, size_t size)) { return dsr::GuestAlignedAlloc(align, size); }

HLE_FUNC(memset_s, int, libc_memset_s, (void* dst, size_t dmax, int c, size_t n)) {
    if (!dst) return EINVAL;
    size_t k = n < dmax ? n : dmax;
    volatile auto* p = static_cast<volatile unsigned char*>(dst);
    for (size_t i = 0; i < k; i++) p[i] = static_cast<unsigned char>(c);
    return n > dmax ? EOVERFLOW : 0;
}

// qsort/bsearch: callbacks are guest functions.
namespace {
thread_local const void* t_cmp = nullptr;
int GuestCompare(const void* a, const void* b) { return dsr::arch::GuestCall<int>(t_cmp, a, b); }
}  // namespace

HLE_FUNC(qsort, void, libc_qsort, (void* base, size_t n, size_t sz, const void* cmp)) {
    const void* saved = t_cmp;
    t_cmp = cmp;
    qsort(base, n, sz, GuestCompare);
    t_cmp = saved;
}

HLE_FUNC(bsearch, void*, libc_bsearch, (const void* key, const void* base, size_t n, size_t sz, const void* cmp)) {
    const void* saved = t_cmp;
    t_cmp = cmp;
    void* r = bsearch(key, base, n, sz, GuestCompare);
    t_cmp = saved;
    return r;
}

HLE_FUNC(atexit, int, libc_atexit, (const void* fn)) {
    std::lock_guard<std::mutex> lk(g_atexit_mutex);
    g_atexit.push_back(fn);
    return 0;
}

HLE_FUNC(exit, void, libc_exit, (int code)) {
    LOGW(kTag, "guest called exit(%d)", code);
    exit(code);
}

HLE_FUNC(abort, void, libc_abort, ()) { FATAL(kTag, "guest called abort()"); }

HLE_FUNC(__cxa_pure_virtual, void, libc_cxa_pure_virtual, ()) { FATAL(kTag, "guest called a pure virtual function"); }

HLE_ALIAS(setjmp, dsr_setjmp);
HLE_ALIAS(longjmp, dsr_longjmp);

// musl init/fini hooks used by main's _init/_fini. Returning 0 makes _init run DT_INIT_ARRAY itself.
HLE_FUNC(__nnmusl_init_dso, int, libc_nnmusl_init_dso, ()) { return 0; }
HLE_FUNC(__nnmusl_fini_dso, void, libc_nnmusl_fini_dso, ()) {}

// ---- nn::Result C helpers ------------------------------------------------------------------------
HLE_FUNC(nnResultIsSuccess, bool, nnResultIsSuccess, (uint32_t r)) { return r == 0; }
HLE_FUNC(nnResultIsFailure, bool, nnResultIsFailure, (uint32_t r)) { return r != 0; }
HLE_FUNC(nnResultGetModule, int, nnResultGetModule, (uint32_t r)) { return int(r & 0x1FF); }
HLE_FUNC(nnResultGetDescription, int, nnResultGetDescription, (uint32_t r)) { return int((r >> 9) & 0x1FFF); }
