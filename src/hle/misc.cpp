// Small nn:: services: oe, settings, time, diag, util, err, account (single local user), nv.
#include <time.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "common/log.h"
#include "hle/registry.h"
#include "runtime/runtime.h"

namespace {
constexpr uint32_t kOk = 0;

// ---- languages ---------------------------------------------------------------------------------------
// nn::settings::Language enum order -> LanguageCode string.
const char* const kLanguageCodes[] = {"ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl",
                                      "pt", "ru", "zh-TW", "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant"};
constexpr int kNumLanguages = int(sizeof(kLanguageCodes) / sizeof(kLanguageCodes[0]));

uint64_t PackCode(const char* s) {
    uint64_t v = 0;
    memcpy(&v, s, strnlen(s, 8));
    return v;
}
}  // namespace

// ===== nn::oe ======================================================================================
HLE_FUNC(_ZN2nn2oe10InitializeEv, void, oe_Initialize, ()) {}
HLE_FUNC(_ZN2nn2oe16GetOperationModeEv, int, oe_GetOperationMode, ()) { return dsr::GetRuntime().config.docked ? 1 : 0; }
HLE_FUNC(_ZN2nn2oe20GetCurrentFocusStateEv, int, oe_GetCurrentFocusState, ()) { return 1; /* InFocus */ }
HLE_FUNC(_ZN2nn2oe18GetDesiredLanguageEv, uint64_t, oe_GetDesiredLanguage, ()) {
    return PackCode(dsr::GetRuntime().config.language.c_str());
}
HLE_FUNC(_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE, void, oe_GetDisplayVersion, (char* out)) {
    strncpy(out, "1.0.3", 16);
}
HLE_FUNC(_ZN2nn2oe25TryPopNotificationMessageEPj, bool, oe_TryPopNotificationMessage, (uint32_t*)) { return false; }
HLE_FUNC(_ZN2nn2oe20SetFocusHandlingModeENS0_17FocusHandlingModeE, void, oe_SetFocusHandlingMode, (int)) {}
HLE_FUNC(_ZN2nn2oe28SetResumeNotificationEnabledEb, void, oe_SetResumeNotificationEnabled, (bool)) {}
HLE_FUNC(_ZN2nn2oe42SetOperationModeChangedNotificationEnabledEb, void, oe_SetOpModeNotif, (bool)) {}
HLE_FUNC(_ZN2nn2oe44SetPerformanceModeChangedNotificationEnabledEb, void, oe_SetPerfModeNotif, (bool)) {}
HLE_FUNC(_ZN2nn2oe27SetPerformanceConfigurationENS0_15PerformanceModeEi, void, oe_SetPerformanceConfiguration,
         (int, int)) {}
HLE_FUNC(_ZN2nn2oe31EnterExitRequestHandlingSectionEv, void, oe_EnterExitSection, ()) {}
HLE_FUNC(_ZN2nn2oe31LeaveExitRequestHandlingSectionEv, void, oe_LeaveExitSection, ()) {}
HLE_FUNC(_ZN2nn2oe15EnableRecordingEv, void, oe_EnableRecording, ()) {}
HLE_FUNC(_ZN2nn2oe16DisableRecordingEv, void, oe_DisableRecording, ()) {}
HLE_FUNC(_ZN2nn2oe16EnableScreenShotEv, void, oe_EnableScreenShot, ()) {}
HLE_FUNC(_ZN2nn2oe17DisableScreenShotEv, void, oe_DisableScreenShot, ()) {}

// ===== nn::settings ================================================================================
HLE_FUNC(_ZN2nn8settings12LanguageCode4MakeENS0_8LanguageE, uint64_t, settings_LanguageCodeMake, (int lang)) {
    return PackCode(lang >= 0 && lang < kNumLanguages ? kLanguageCodes[lang] : "en-US");
}
HLE_FUNC(_ZN2nn8settingseqERKNS0_12LanguageCodeES3_, bool, settings_LanguageCodeEq,
         (const uint64_t* a, const uint64_t* b)) {
    return *a == *b;
}
HLE_FUNC(_ZN2nn8settings29GetAvailableLanguageCodeCountEv, int, settings_GetAvailableLanguageCodeCount, ()) {
    return 15;  // SDK 4.x exposes the first 15 languages
}
HLE_FUNC(_ZN2nn8settings25GetAvailableLanguageCodesEPNS0_12LanguageCodeEi, int, settings_GetAvailableLanguageCodes,
         (uint64_t * out, int count)) {
    int n = count < 15 ? count : 15;
    for (int i = 0; i < n; i++) out[i] = PackCode(kLanguageCodes[i]);
    return n;
}

// ===== nn::time ====================================================================================
namespace {
struct CalendarTime {
    int16_t year;
    int8_t month, day, hour, minute, second;
    int8_t pad;
};
struct CalendarAdditionalInfo {
    int32_t day_of_week;
    int32_t day_of_year;
    char tz_name[8];
    bool is_dst;
    int32_t utc_offset_seconds;
};
struct SteadyClockTimePoint {
    int64_t value;
    uint8_t source_id[16];
};
CalendarTime FromTm(const tm& t) {
    return {int16_t(t.tm_year + 1900), int8_t(t.tm_mon + 1), int8_t(t.tm_mday), int8_t(t.tm_hour),
            int8_t(t.tm_min), int8_t(t.tm_sec), 0};
}
}  // namespace

HLE_FUNC(_ZN2nn4time10InitializeEv, uint32_t, time_Initialize, ()) { return kOk; }
HLE_FUNC(_ZN2nn4time8FinalizeEv, uint32_t, time_Finalize, ()) { return kOk; }
HLE_FUNC(_ZN2nn4time13IsInitializedEv, bool, time_IsInitialized, ()) { return true; }
HLE_FUNC(_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE, uint32_t, time_UserGetCurrentTime,
         (int64_t * out)) {
    *out = int64_t(::time(nullptr));
    return kOk;
}
HLE_FUNC(_ZN2nn4time26StandardNetworkSystemClock14GetCurrentTimeEPNS0_9PosixTimeE, uint32_t,
         time_NetworkGetCurrentTime, (int64_t * out)) {
    *out = int64_t(::time(nullptr));
    return kOk;
}
HLE_FUNC(_ZN2nn4time19StandardSteadyClock3nowEv, SteadyClockTimePoint, time_SteadyNow, ()) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    SteadyClockTimePoint p{};
    p.value = ts.tv_sec;
    return p;
}
HLE_FUNC(_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE, uint32_t,
         time_ToCalendarTime, (CalendarTime * out, CalendarAdditionalInfo* info, const int64_t* posix)) {
    time_t t = time_t(*posix);
    tm lt;
    localtime_r(&t, &lt);
    *out = FromTm(lt);
    if (info) {
        memset(info, 0, sizeof(*info));
        info->day_of_week = lt.tm_wday;
        info->day_of_year = lt.tm_yday;
        strncpy(info->tz_name, lt.tm_zone ? lt.tm_zone : "UTC", sizeof(info->tz_name));
        info->is_dst = lt.tm_isdst > 0;
        info->utc_offset_seconds = int32_t(lt.tm_gmtoff);
    }
    return kOk;
}
HLE_FUNC(_ZN2nn4time19ToCalendarTimeInUtcERKNS0_9PosixTimeE, CalendarTime, time_ToCalendarTimeInUtc,
         (const int64_t* posix)) {
    time_t t = time_t(*posix);
    tm g;
    gmtime_r(&t, &g);
    return FromTm(g);
}
HLE_FUNC(_ZN2nn4time11ToPosixTimeEPiPNS0_9PosixTimeEiRKNS0_12CalendarTimeE, uint32_t, time_ToPosixTime,
         (int* out_count, int64_t* out, int count, const CalendarTime* c)) {
    tm t{};
    t.tm_year = c->year - 1900;
    t.tm_mon = c->month - 1;
    t.tm_mday = c->day;
    t.tm_hour = c->hour;
    t.tm_min = c->minute;
    t.tm_sec = c->second;
    t.tm_isdst = -1;
    if (count > 0) out[0] = int64_t(mktime(&t));
    if (out_count) *out_count = count > 0 ? 1 : 0;
    return kOk;
}

// ===== nn::diag ====================================================================================
HLE_FUNC(_ZN2nn4diag6detail9AbortImplEPKcS3_S3_i, void, diag_Abort,
         (const char* cond, const char* func, const char* file, int line)) {
    FATAL("guest", "abort: %s in %s (%s:%d)", cond ? cond : "", func ? func : "", file ? file : "", line);
}
HLE_FUNC(_ZN2nn4diag6detail9AbortImplEPKcS3_S3_iPKNS_6ResultES3_z, void, diag_AbortFmt,
         (const char* cond, const char* func, const char* file, int line, const uint32_t* result, const char* fmt,
          ...)) {
    char msg[1024] = "";
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
    }
    FATAL("guest", "abort: %s in %s (%s:%d) result=0x%x %s", cond ? cond : "", func ? func : "", file ? file : "",
          line, result ? *result : 0, msg);
}
HLE_FUNC(_ZN2nn4diag6detail18OnAssertionFailureENS0_13AssertionTypeEPKcS4_S4_i, void, diag_OnAssertionFailure,
         (int type, const char* cond, const char* func, const char* file, int line)) {
    LOGE("guest", "assertion(%d) failed: %s in %s (%s:%d)", type, cond ? cond : "", func ? func : "",
         file ? file : "", line);
}
HLE_FUNC(_ZN2nn4diag6detail7LogImplERKNS0_11LogMetaDataEPKcz, void, diag_LogImpl, (const void*, const char* fmt, ...)) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    size_t n = strlen(msg);
    while (n && msg[n - 1] == '\n') msg[--n] = 0;
    LOGI("guest", "%s", msg);
}
HLE_FUNC(_ZN2nn6detail21UnexpectedDefaultImplEPKcS2_i, void, detail_UnexpectedDefault,
         (const char* func, const char* file, int line)) {
    FATAL("guest", "unexpected default in %s (%s:%d)", func ? func : "", file ? file : "", line);
}
HLE_FUNC(nndetailDiagAbortFormatImpl, void, nndetailDiagAbortFormatImpl, ()) { FATAL("guest", "nn diag abort"); }

// ===== nn::util ====================================================================================
HLE_FUNC(_ZN2nn4util8SNPrintfEPcmPKcz, int, util_SNPrintf, (char* buf, size_t n, const char* fmt, ...)) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf, n, fmt, ap);
    va_end(ap);
    return r;
}
HLE_FUNC(_ZN2nn4util9VSNPrintfEPcmPKcSt9__va_list, int, util_VSNPrintf,
         (char* buf, size_t n, const char* fmt, va_list ap)) {
    return vsnprintf(buf, n, fmt, ap);
}
HLE_FUNC(_ZN2nn4util11ReferSymbolEPKv, void, util_ReferSymbol, (const void*)) {}

// ===== nn::err =====================================================================================
// ErrorResultVariant (12 bytes, copied by value by the game): we store {kind, a, b}:
//   kind 1 = nn::Result (a = result), kind 2 = ErrorCode (a = category, b = number).
// Each distinct error is logged once (with a repeat count every 10th time) and reported with the guest
// address that raised it; on a real console these are modal error dialogs.
namespace {
std::mutex g_err_mutex;
std::map<std::string, int> g_err_counts;

void ReportError(const char* what, uintptr_t caller) {
    int n;
    {
        std::lock_guard<std::mutex> lk(g_err_mutex);
        n = ++g_err_counts[what];
    }
    const auto& m = dsr::GetRuntime().main;
    unsigned long off = m.Contains(caller) ? (unsigned long)(caller - m.base()) : 0;
    if (n == 1 || n % 10 == 0)
        LOGW("err", "guest error dialog: %s (shown %d time%s, from main+0x%lx)", what, n, n == 1 ? "" : "s", off);
}
}  // namespace

HLE_FUNC(_ZN2nn3err9ShowErrorENS_6ResultE, void, err_ShowError, (uint32_t r)) {
    char buf[96];
    snprintf(buf, sizeof(buf), "result 0x%x (module %u, description %u)", r, r & 0x1FF, (r >> 9) & 0x1FFF);
    ReportError(buf, reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
}
HLE_FUNC(_ZN2nn3err9ShowErrorERKNS0_18ErrorResultVariantE, void, err_ShowErrorVariant, (const uint32_t* v)) {
    char buf[96];
    if (v[0] == 1)
        snprintf(buf, sizeof(buf), "result 0x%x (module %u, description %u)", v[1], v[1] & 0x1FF, (v[1] >> 9) & 0x1FFF);
    else if (v[0] == 2)
        snprintf(buf, sizeof(buf), "error code %04u-%04u", v[1], v[2]);
    else
        snprintf(buf, sizeof(buf), "empty error");
    ReportError(buf, reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
}
HLE_FUNC(_ZN2nn3err18ErrorResultVariantC1Ev, void, err_ERV_ctor, (uint32_t * self)) {
    self[0] = self[1] = self[2] = 0;
}
HLE_FUNC(_ZN2nn3err18ErrorResultVariantC1ERKNS_6ResultE, void, err_ERV_ctor_result,
         (uint32_t * self, const uint32_t* r)) {
    self[0] = 1;
    self[1] = *r;
    self[2] = 0;
}
HLE_FUNC(_ZN2nn3err18ErrorResultVariantC1ERKNS0_9ErrorCodeE, void, err_ERV_ctor_code,
         (uint32_t * self, const uint32_t* code)) {
    self[0] = 2;
    self[1] = code[0];
    self[2] = code[1];
}
HLE_FUNC(_ZN2nn3err9ErrorCode19GetInvalidErrorCodeEv, uint64_t, err_GetInvalidErrorCode, ()) { return 0; }
HLE_FUNC(_ZNK2nn3err9ErrorCode7IsValidEv, bool, err_ErrorCodeIsValid, (const uint32_t* self)) {
    return self[0] != 0 || self[1] != 0;
}

// ===== nn::account — one local user, no network service account ===================================
namespace {
constexpr uint64_t kUid[2] = {0x0123456789ABCDEFull, 0xFEDCBA9876543210ull};
constexpr uint32_t kAccountNsaUnavailable = 124u | (280u << 9);  // module 124 (account); game only checks failure
}  // namespace

HLE_FUNC(_ZN2nn7account10InitializeEv, void, account_Initialize, ()) {}
HLE_FUNC(_ZN2nn7account12ListAllUsersEPiPNS0_3UidEi, uint32_t, account_ListAllUsers,
         (int* count, uint64_t* uids, int max)) {
    *count = 0;
    if (max > 0) {
        uids[0] = kUid[0];
        uids[1] = kUid[1];
        *count = 1;
    }
    return kOk;
}
HLE_ALIAS(_ZN2nn7account13ListOpenUsersEPiPNS0_3UidEi, account_ListAllUsers);
HLE_FUNC(_ZN2nn7account19OpenPreselectedUserEPNS0_10UserHandleE, uint32_t, account_OpenPreselectedUser,
         (uint64_t * handle)) {
    handle[0] = kUid[0];
    handle[1] = kUid[1];
    return kOk;
}
HLE_FUNC(_ZN2nn7account8OpenUserEPNS0_10UserHandleERKNS0_3UidE, uint32_t, account_OpenUser,
         (uint64_t * handle, const uint64_t* uid)) {
    handle[0] = uid[0];
    handle[1] = uid[1];
    return kOk;
}
HLE_FUNC(_ZN2nn7account9CloseUserERKNS0_10UserHandleE, void, account_CloseUser, (const void*)) {}
HLE_FUNC(_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE, uint32_t, account_GetUserId,
         (uint64_t * uid, const uint64_t* handle)) {
    uid[0] = handle[0];
    uid[1] = handle[1];
    return kOk;
}
HLE_FUNC(_ZN2nn7account17GetLastOpenedUserEPNS0_3UidE, uint32_t, account_GetLastOpenedUser, (uint64_t * uid)) {
    uid[0] = kUid[0];
    uid[1] = kUid[1];
    return kOk;
}
HLE_FUNC(_ZN2nn7account16GetUserExistenceEPbRKNS0_3UidE, uint32_t, account_GetUserExistence,
         (bool* out, const uint64_t* uid)) {
    *out = uid[0] == kUid[0] && uid[1] == kUid[1];
    return kOk;
}
HLE_FUNC(_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE, uint32_t, account_GetNickname,
         (char* out, const void*)) {
    strncpy(out, "Chosen Undead", 33);
    return kOk;
}
HLE_FUNC(_ZN2nn7account32IsNetworkServiceAccountAvailableEPbRKNS0_10UserHandleE, uint32_t,
         account_IsNsaAvailable, (bool* out, const void*)) {
    *out = false;
    return kOk;
}
HLE_FUNC(_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE, uint32_t, account_EnsureNsa,
         (const void*)) {
    return kAccountNsaUnavailable;
}
HLE_FUNC(_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE, uint32_t,
         account_GetNsaId, (uint64_t * out, const void*)) {
    *out = 0;
    return kAccountNsaUnavailable;
}

// ===== nv (graphics driver memory setup) ===========================================================
HLE_FUNC(_ZN2nv18InitializeGraphicsEPvm, void, nv_InitializeGraphics, (void* mem, size_t size)) {
    LOGI("nv", "InitializeGraphics(%p, %zu MiB)", mem, size >> 20);
}
HLE_FUNC(_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_, void, nv_SetGraphicsAllocator,
         (const void*, const void*, const void*, void*)) {}
HLE_FUNC(_ZN2nv28SetGraphicsDevtoolsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_, void,
         nv_SetGraphicsDevtoolsAllocator, (const void*, const void*, const void*, void*)) {}
