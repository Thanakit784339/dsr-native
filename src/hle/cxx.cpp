// C++ runtime imports (Nintendo's libc++, inline namespace std::__1).
//
// The NDK's libc++ uses the same ABI version (v1) as the Switch build, so std::string, iostream and
// locale objects have identical layouts and we can operate on guest objects with host code.
// (On non-Android hosts these wrappers compile against whatever standard library is present; they
// are never executed in dry-run builds.)
//
// Allocation: main exports malloc/free, so guest `operator new` must use the game allocator. Host
// code keeps using the host heap. Every `operator delete` (guest-facing and the host's global one)
// dispatches on the pointer's address, so memory can cross the boundary in either direction (e.g.
// a host std::string::append reallocating a buffer the game allocated inline).

#include <time.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <istream>
#include <locale>
#include <mutex>
#include <new>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/log.h"
#include "hle/os/heap.h"
#include "hle/registry.h"
#include "runtime/runtime.h"

using dsr::hle::os::GuestHeapContains;

namespace {
constexpr const char* kTag = "cxx";

void DispatchFree(void* p) {
    if (!p) return;
    if (GuestHeapContains(p))
        dsr::GuestFree(p);
    else
        free(p);
}
}  // namespace

// ---- host global operator new/delete (affects every C++ allocation in this library) -------------
void* operator new(size_t n) {
    void* p = malloc(n ? n : 1);
    if (!p) FATAL(kTag, "host operator new(%zu) failed", n);
    return p;
}
void* operator new[](size_t n) { return operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept { return malloc(n ? n : 1); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return malloc(n ? n : 1); }
void operator delete(void* p) noexcept { DispatchFree(p); }
void operator delete[](void* p) noexcept { DispatchFree(p); }
void operator delete(void* p, size_t) noexcept { DispatchFree(p); }
void operator delete[](void* p, size_t) noexcept { DispatchFree(p); }

// ---- guest-facing operator new/delete ------------------------------------------------------------
HLE_FUNC(_Znwm, void*, cxx_new, (size_t n)) {
    void* p = dsr::GuestMalloc(n ? n : 1);
    if (!p) FATAL(kTag, "guest operator new(%zu) failed", n);
    return p;
}
HLE_ALIAS(_Znam, cxx_new);
HLE_FUNC(_ZnwmRKSt9nothrow_t, void*, cxx_new_nothrow, (size_t n, const void*)) { return dsr::GuestMalloc(n ? n : 1); }
HLE_FUNC(_ZdlPv, void, cxx_delete, (void* p)) { DispatchFree(p); }
HLE_ALIAS(_ZdaPv, cxx_delete);

static char g_nothrow;
HLE_DATA(_ZSt7nothrow, g_nothrow);

// ---- libc++ internal error paths -------------------------------------------------------------------
HLE_FUNC(_ZNKSt3__120__vector_base_commonILb1EE20__throw_length_errorEv, void, cxx_vec_len, (const void*)) {
    FATAL(kTag, "std::vector length_error");
}
HLE_FUNC(_ZNKSt3__120__vector_base_commonILb1EE20__throw_out_of_rangeEv, void, cxx_vec_oor, (const void*)) {
    FATAL(kTag, "std::vector out_of_range");
}
HLE_FUNC(_ZNKSt3__121__basic_string_commonILb1EE20__throw_length_errorEv, void, cxx_str_len, (const void*)) {
    FATAL(kTag, "std::string length_error");
}
HLE_FUNC(_ZNKSt3__121__basic_string_commonILb1EE20__throw_out_of_rangeEv, void, cxx_str_oor, (const void*)) {
    FATAL(kTag, "std::string out_of_range");
}

// ---- std::string -----------------------------------------------------------------------------------
using Str = std::string;
static Str* S(void* p) { return static_cast<Str*>(p); }

HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEPKcm, void, str_init,
         (void* self, const char* s, size_t n)) {
    new (self) Str(s, n);
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_, void, str_copy_ctor,
         (void* self, const void* other)) {
    new (self) Str(*static_cast<const Str*>(other));
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1ERKS5_mmRKS4_, void, str_sub_ctor,
         (void* self, const void* other, size_t pos, size_t n, const void*)) {
    new (self) Str(*static_cast<const Str*>(other), pos, n);
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev, void, str_dtor, (void* self)) {
    S(self)->~Str();
}
HLE_ALIAS(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED2Ev, str_dtor);
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEaSERKS5_, void*, str_assign_op,
         (void* self, const void* other)) {
    *S(self) = *static_cast<const Str*>(other);
    return self;
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6assignEPKc, void*, str_assign,
         (void* self, const char* s)) {
    S(self)->assign(s);
    return self;
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKcm, void*, str_append_n,
         (void* self, const char* s, size_t n)) {
    S(self)->append(s, n);
    return self;
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6appendEPKc, void*, str_append,
         (void* self, const char* s)) {
    S(self)->append(s);
    return self;
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE9push_backEc, void, str_push_back,
         (void* self, char c)) {
    S(self)->push_back(c);
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7reserveEm, void, str_reserve,
         (void* self, size_t n)) {
    S(self)->reserve(n);
}
HLE_FUNC(_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6resizeEmc, void, str_resize,
         (void* self, size_t n, char c)) {
    S(self)->resize(n, c);
}
HLE_FUNC(_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEPKc, int, str_compare,
         (const void* self, const char* s)) {
    return static_cast<const Str*>(self)->compare(s);
}
HLE_FUNC(_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE7compareEmmPKcm, int, str_compare_sub,
         (const void* self, size_t pos, size_t n1, const char* s, size_t n2)) {
    return static_cast<const Str*>(self)->compare(pos, n1, s, n2);
}
HLE_FUNC(_ZNSt3__14stoiERKNS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEPmi, int, str_stoi,
         (const void* s, size_t* idx, int base)) {
    const Str& str = *static_cast<const Str*>(s);
    char* end = nullptr;
    long v = strtol(str.c_str(), &end, base);
    if (idx) *idx = size_t(end - str.c_str());
    return int(v);
}
// Returns std::string by value: indirect result in x8 on both sides (non-trivial class).
HLE_FUNC(_ZNSt3__19to_stringEj, Str, str_to_string, (unsigned v)) { return std::to_string(v); }

// ---- misc algorithms / helpers ---------------------------------------------------------------------
HLE_FUNC(_ZNSt3__16__sortIRNS_6__lessIiiEEPiEEvT0_S5_T_, void, cxx_sort_int, (int* first, int* last, void*)) {
    std::sort(first, last);
}

HLE_FUNC(_ZNSt3__112__next_primeEm, size_t, cxx_next_prime, (size_t n)) {
    if (n <= 2) return 2;
    if ((n & 1) == 0) n++;
    for (;; n += 2) {
        bool prime = true;
        for (size_t d = 3; d * d <= n; d += 2)
            if (n % d == 0) {
                prime = false;
                break;
            }
        if (prime) return n;
    }
}

// std::__rs_default (random_shuffle RNG) — empty object, operator() returns a 32-bit value.
HLE_FUNC(_ZNSt3__18__rs_getEv, void, cxx_rs_get, ()) {}
HLE_FUNC(_ZNSt3__112__rs_defaultD1Ev, void, cxx_rs_dtor, (void*)) {}
HLE_FUNC(_ZNSt3__112__rs_defaultclEv, uint32_t, cxx_rs_call, (void*)) {
    static std::mt19937 rng(0x5EED);
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    return rng();
}

// std::random_device: guest object is { int fd }; we ignore it.
HLE_FUNC(_ZNSt3__113random_deviceC1ERKNS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE, void,
         cxx_rd_ctor, (void* self, const void*)) {
    *static_cast<int*>(self) = -1;
}
HLE_FUNC(_ZNSt3__113random_deviceD1Ev, void, cxx_rd_dtor, (void*)) {}
HLE_FUNC(_ZNSt3__113random_deviceclEv, uint32_t, cxx_rd_call, (void*)) {
    static std::random_device rd;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    return rd();
}

// chrono: libc++ steady_clock ticks in ns, system_clock in µs.
HLE_FUNC(_ZNSt3__16chrono12steady_clock3nowEv, int64_t, cxx_steady_now, ()) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}
HLE_FUNC(_ZNSt3__16chrono12system_clock3nowEv, int64_t, cxx_system_now, ()) {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return int64_t(ts.tv_sec) * 1'000'000 + ts.tv_nsec / 1000;
}
HLE_FUNC(_ZNSt3__111this_thread9sleep_forERKNS_6chrono8durationIxNS_5ratioILl1ELl1000000000EEEEE, void,
         cxx_sleep_for, (const int64_t* ns)) {
    if (*ns <= 0) return;
    timespec ts{time_t(*ns / 1'000'000'000), long(*ns % 1'000'000'000)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {
    }
}

// ---- std::mutex / recursive_mutex / condition_variable ---------------------------------------------
// Guest objects wrap Nintendo pthread types whose size/initializer differ from bionic, so host objects
// are kept in a table keyed by the guest address and created on first use.
namespace {
template <typename T>
class AddressTable {
public:
    T* Get(void* key) {
        Shard& s = shards_[(reinterpret_cast<uintptr_t>(key) >> 4) % kShards];
        std::lock_guard<std::mutex> lk(s.m);
        auto& slot = s.map[key];
        if (!slot) slot = new T;
        return slot;
    }
    void Erase(void* key) {
        Shard& s = shards_[(reinterpret_cast<uintptr_t>(key) >> 4) % kShards];
        std::lock_guard<std::mutex> lk(s.m);
        auto it = s.map.find(key);
        if (it != s.map.end()) {
            delete it->second;
            s.map.erase(it);
        }
    }

private:
    static constexpr size_t kShards = 64;
    struct Shard {
        std::mutex m;
        std::unordered_map<void*, T*> map;
    };
    Shard shards_[kShards];
};

AddressTable<std::mutex>& Mutexes() {
    static auto* t = new AddressTable<std::mutex>;
    return *t;
}
AddressTable<std::recursive_mutex>& RecMutexes() {
    static auto* t = new AddressTable<std::recursive_mutex>;
    return *t;
}
AddressTable<std::condition_variable>& CondVars() {
    static auto* t = new AddressTable<std::condition_variable>;
    return *t;
}

struct GuestUniqueLock {
    void* mutex;
    bool owns;
};
}  // namespace

HLE_FUNC(_ZNSt3__15mutex4lockEv, void, cxx_mutex_lock, (void* m)) { Mutexes().Get(m)->lock(); }
HLE_FUNC(_ZNSt3__15mutex6unlockEv, void, cxx_mutex_unlock, (void* m)) { Mutexes().Get(m)->unlock(); }
HLE_FUNC(_ZNSt3__15mutexD1Ev, void, cxx_mutex_dtor, (void* m)) { Mutexes().Erase(m); }

HLE_FUNC(_ZNSt3__115recursive_mutexC1Ev, void, cxx_rmutex_ctor, (void* m)) { RecMutexes().Get(m); }
HLE_FUNC(_ZNSt3__115recursive_mutexD1Ev, void, cxx_rmutex_dtor, (void* m)) { RecMutexes().Erase(m); }
HLE_FUNC(_ZNSt3__115recursive_mutex4lockEv, void, cxx_rmutex_lock, (void* m)) { RecMutexes().Get(m)->lock(); }
HLE_FUNC(_ZNSt3__115recursive_mutex6unlockEv, void, cxx_rmutex_unlock, (void* m)) { RecMutexes().Get(m)->unlock(); }

HLE_FUNC(_ZNSt3__118condition_variable10notify_allEv, void, cxx_cv_notify_all, (void* cv)) {
    CondVars().Get(cv)->notify_all();
}
HLE_FUNC(_ZNSt3__118condition_variableD1Ev, void, cxx_cv_dtor, (void* cv)) { CondVars().Erase(cv); }
HLE_FUNC(_ZNSt3__118condition_variable4waitERNS_11unique_lockINS_5mutexEEE, void, cxx_cv_wait,
         (void* cv, GuestUniqueLock* lk)) {
    std::unique_lock<std::mutex> hl(*Mutexes().Get(lk->mutex), std::adopt_lock);
    CondVars().Get(cv)->wait(hl);
    hl.release();
}
HLE_FUNC(
    _ZNSt3__118condition_variable15__do_timed_waitERNS_11unique_lockINS_5mutexEEENS_6chrono10time_pointINS5_12system_clockENS5_8durationIxNS_5ratioILl1ELl1000000000EEEEEEE,
    void, cxx_cv_timed_wait, (void* cv, GuestUniqueLock* lk, int64_t abs_ns_since_epoch)) {
    std::unique_lock<std::mutex> hl(*Mutexes().Get(lk->mutex), std::adopt_lock);
    auto tp = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::nanoseconds(abs_ns_since_epoch)));
    CondVars().Get(cv)->wait_until(hl, tp);
    hl.release();
}

// ---- iostreams / locale ---------------------------------------------------------------------------
namespace {
// Accessors for protected members, called non-virtually on guest objects.
struct SB : std::streambuf {
    static int Sync(void* s) { return static_cast<SB*>(s)->std::streambuf::sync(); }
    static void Imbue(void* s, const std::locale& l) { static_cast<SB*>(s)->std::streambuf::imbue(l); }
    static int Uflow(void* s) { return static_cast<SB*>(s)->std::streambuf::uflow(); }
    static std::streambuf* Setbuf(void* s, char* b, std::streamsize n) {
        return static_cast<SB*>(s)->std::streambuf::setbuf(b, n);
    }
    static std::streamsize Xsgetn(void* s, char* b, std::streamsize n) {
        return static_cast<SB*>(s)->std::streambuf::xsgetn(b, n);
    }
    static std::streamsize Xsputn(void* s, const char* b, std::streamsize n) {
        return static_cast<SB*>(s)->std::streambuf::xsputn(b, n);
    }
    static std::streamsize Showmanyc(void* s) { return static_cast<SB*>(s)->std::streambuf::showmanyc(); }
};
struct IOS : std::ios {
    static void Init(void* ios, void* sb) { static_cast<IOS*>(ios)->std::ios::init(static_cast<std::streambuf*>(sb)); }
};
}  // namespace

HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEEC2Ev, void, sb_ctor, (void* self)) {
    new (self) SB();  // guest constructor installs its own vtable afterwards
}
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEED2Ev, void, sb_dtor, (void* self)) {
    static_cast<SB*>(self)->std::streambuf::~basic_streambuf();
}
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE4syncEv, int, sb_sync, (void* s)) { return SB::Sync(s); }
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5imbueERKNS_6localeE, void, sb_imbue,
         (void* s, const std::locale* l)) {
    SB::Imbue(s, *l);
}
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5uflowEv, int, sb_uflow, (void* s)) { return SB::Uflow(s); }
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE6setbufEPcl, void*, sb_setbuf,
         (void* s, char* b, long n)) {
    return SB::Setbuf(s, b, n);
}
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE6xsgetnEPcl, long, sb_xsgetn, (void* s, char* b, long n)) {
    return SB::Xsgetn(s, b, n);
}
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE6xsputnEPKcl, long, sb_xsputn,
         (void* s, const char* b, long n)) {
    return SB::Xsputn(s, b, n);
}
HLE_FUNC(_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE9showmanycEv, long, sb_showmanyc, (void* s)) {
    return SB::Showmanyc(s);
}

HLE_FUNC(_ZNSt3__18ios_base4initEPv, void, ios_init, (void* ios, void* sb)) { IOS::Init(ios, sb); }
HLE_FUNC(_ZNSt3__18ios_base5clearEj, void, ios_clear, (void* ios, unsigned st)) {
    static_cast<std::ios*>(ios)->clear(std::ios::iostate(st));
}
HLE_FUNC(_ZNKSt3__18ios_base6getlocEv, std::locale, ios_getloc, (const void* ios)) {
    return static_cast<const std::ios_base*>(ios)->getloc();
}
HLE_FUNC(_ZNSt3__19basic_iosIcNS_11char_traitsIcEEED2Ev, void, ios_dtor, (void* ios)) {
    static_cast<std::ios*>(ios)->std::ios::~basic_ios();
}

HLE_FUNC(_ZNSt3__16localeC1ERKS0_, void, locale_copy, (void* self, const std::locale* other)) {
    new (self) std::locale(*other);
}
HLE_FUNC(_ZNSt3__16localeD1Ev, void, locale_dtor, (void* self)) { static_cast<std::locale*>(self)->~locale(); }
HLE_FUNC(_ZNKSt3__16locale9use_facetERNS0_2idE, const void*, locale_use_facet,
         (const std::locale* self, std::locale::id* id)) {
    if (id == &std::ctype<char>::id) return &std::use_facet<std::ctype<char>>(*self);
    if (id == &std::codecvt<char, char, std::mbstate_t>::id)
        return &std::use_facet<std::codecvt<char, char, std::mbstate_t>>(*self);
    FATAL(kTag, "locale::use_facet: unsupported facet id %p", static_cast<void*>(id));
}
HLE_FUNC(_ZNKSt3__16locale9has_facetERNS0_2idE, bool, locale_has_facet,
         (const std::locale* self, std::locale::id* id)) {
    if (id == &std::ctype<char>::id) return std::has_facet<std::ctype<char>>(*self);
    if (id == &std::codecvt<char, char, std::mbstate_t>::id)
        return std::has_facet<std::codecvt<char, char, std::mbstate_t>>(*self);
    return false;
}
HLE_DATA(_ZNSt3__15ctypeIcE2idE, std::ctype<char>::id);
HLE_DATA(_ZNSt3__17codecvtIcc11__mbstate_tE2idE, (std::codecvt<char, char, std::mbstate_t>::id));

HLE_DATA(_ZNSt3__14coutE, std::cout);
HLE_DATA(_ZNSt3__14cerrE, std::cerr);
HLE_DATA(_ZNSt3__15wcoutE, std::wcout);
HLE_DATA(_ZNSt3__15wcerrE, std::wcerr);

HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5flushEv, void*, os_flush, (std::ostream * os)) {
    return &os->flush();
}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5writeEPKcl, void*, os_write,
         (std::ostream * os, const char* s, long n)) {
    return &os->write(s, n);
}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE3putEc, void*, os_put, (std::ostream * os, char c)) {
    return &os->put(c);
}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEi, void*, os_put_int, (std::ostream * os, int v)) {
    return &(*os << v);
}
HLE_FUNC(_ZNSt3__113basic_ostreamIwNS_11char_traitsIwEEE5flushEv, void*, wos_flush, (std::wostream * os)) {
    return &os->flush();
}
HLE_FUNC(_ZNSt3__113basic_ostreamIwNS_11char_traitsIwEEE5writeEPKwl, void*, wos_write,
         (std::wostream * os, const wchar_t* s, long n)) {
    return &os->write(s, n);
}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_, void, os_sentry_ctor,
         (void* self, std::ostream* os)) {
    new (self) std::ostream::sentry(*os);
}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev, void, os_sentry_dtor, (void* self)) {
    static_cast<std::ostream::sentry*>(self)->~sentry();
}
// basic_ostream destructors. D2 (base-object, receives a VTT) has an empty body in libc++.
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEED2Ev, void, os_dtor_base, (void*, const void*)) {}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEED1Ev, void, os_dtor, (std::ostream * os)) {
    os->std::ostream::~basic_ostream();
}
HLE_FUNC(_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEED0Ev, void, os_dtor_delete, (std::ostream * os)) {
    os->std::ostream::~basic_ostream();
    DispatchFree(os);
}

HLE_FUNC(_ZNSt3__113basic_istreamIcNS_11char_traitsIcEEE3getEv, int, is_get, (std::istream * is)) { return is->get(); }
HLE_FUNC(_ZNSt3__113basic_istreamIcNS_11char_traitsIcEEE4peekEv, int, is_peek, (std::istream * is)) {
    return is->peek();
}
HLE_FUNC(_ZNSt3__113basic_istreamIcNS_11char_traitsIcEEE6sentryC1ERS3_b, void, is_sentry_ctor,
         (void* self, std::istream* is, bool noskip)) {
    new (self) std::istream::sentry(*is, noskip);
}
HLE_FUNC(_ZNSt3__113basic_istreamIcNS_11char_traitsIcEEED2Ev, void, is_dtor_base, (void*, const void*)) {}
HLE_FUNC(_ZNSt3__113basic_istreamIcNS_11char_traitsIcEEED1Ev, void, is_dtor, (std::istream * is)) {
    is->std::istream::~basic_istream();
}
HLE_FUNC(_ZNSt3__113basic_istreamIcNS_11char_traitsIcEEED0Ev, void, is_dtor_delete, (std::istream * is)) {
    is->std::istream::~basic_istream();
    DispatchFree(is);
}

// ---- exceptions / RTTI support objects ---------------------------------------------------------------
// The game is built without exceptions and never calls dynamic_cast, but its typeinfo objects point at
// these vtables. Use the host's real ones so anything that does inspect them sees valid data.
extern "C" {
extern char _ZTVN10__cxxabiv117__class_type_infoE[];
extern char _ZTVN10__cxxabiv120__si_class_type_infoE[];
extern char _ZTVN10__cxxabiv121__vmi_class_type_infoE[];
extern char _ZTVSt12length_error[];
}
HLE_DATA(_ZTVN10__cxxabiv117__class_type_infoE, _ZTVN10__cxxabiv117__class_type_infoE);
HLE_DATA(_ZTVN10__cxxabiv120__si_class_type_infoE, _ZTVN10__cxxabiv120__si_class_type_infoE);
HLE_DATA(_ZTVN10__cxxabiv121__vmi_class_type_infoE, _ZTVN10__cxxabiv121__vmi_class_type_infoE);
HLE_DATA(_ZTVSt12length_error, _ZTVSt12length_error);

HLE_FUNC(_ZNSt11logic_errorC2EPKc, void, cxx_logic_error_ctor, (void* self, const char* msg)) {
    new (self) std::logic_error(msg);
}
HLE_FUNC(_ZNKSt11logic_error4whatEv, const char*, cxx_logic_error_what, (const void* self)) {
    return static_cast<const std::logic_error*>(self)->what();
}

// "virtual thunk to ~basic_ostream/~basic_istream": adjust `this` by the vcall offset stored at
// vtable[-3] (n24), then run the complete-object (D1) or deleting (D0) destructor.
namespace {
template <typename T>
T* VirtualThunkAdjust(void* self) {
    char* p = static_cast<char*>(self);
    ptrdiff_t off = *reinterpret_cast<const ptrdiff_t*>(*reinterpret_cast<char**>(p) - 24);
    return reinterpret_cast<T*>(p + off);
}
}  // namespace
HLE_FUNC(_ZTv0_n24_NSt3__113basic_ostreamIcNS_11char_traitsIcEEED1Ev, void, os_vthunk_dtor, (void* self)) {
    os_dtor(VirtualThunkAdjust<std::ostream>(self));
}
HLE_FUNC(_ZTv0_n24_NSt3__113basic_ostreamIcNS_11char_traitsIcEEED0Ev, void, os_vthunk_dtor_delete, (void* self)) {
    os_dtor_delete(VirtualThunkAdjust<std::ostream>(self));
}
HLE_FUNC(_ZTv0_n24_NSt3__113basic_istreamIcNS_11char_traitsIcEEED1Ev, void, is_vthunk_dtor, (void* self)) {
    is_dtor(VirtualThunkAdjust<std::istream>(self));
}
HLE_FUNC(_ZTv0_n24_NSt3__113basic_istreamIcNS_11char_traitsIcEEED0Ev, void, is_vthunk_dtor_delete, (void* self)) {
    is_dtor_delete(VirtualThunkAdjust<std::istream>(self));
}
