// Online / system-applet services, implemented as "offline": network never becomes available, sockets
// fail with ENETDOWN, SSL/BCAT/friends/nsd fail, amiibo finds no reader, the software keyboard is
// cancelled, and the profanity filter masks nothing. The game's own offline paths take over.
#include <arpa/inet.h>
#include <netinet/in.h>

#include <cerrno>
#include <cstring>

#include <mutex>
#include <string>
#include <algorithm>
#include "common/platform.h"
#include "arch/arch.h"
#include "common/log.h"
#include "hle/os/os.h"
#include "hle/registry.h"

namespace {
constexpr uint32_t kOk = 0;
constexpr uint32_t Fail(uint32_t module, uint32_t desc) { return module | (desc << 9); }
constexpr uint32_t kNifmNoNetwork = Fail(110, 3400);
constexpr uint32_t kSslUnavailable = Fail(123, 1);
constexpr uint32_t kBcatUnavailable = Fail(122, 1);
constexpr uint32_t kNsdUnavailable = Fail(141, 1);
constexpr uint32_t kNfpNeedRestart = Fail(115, 64);   // device not found
constexpr uint32_t kSwkbdCanceled = Fail(140, 1);
constexpr int kNetDown = ENETDOWN;
}  // namespace

// ===== nifm =========================================================================================
HLE_FUNC(_ZN2nn4nifm10InitializeEv, uint32_t, nifm_Initialize, ()) { return kOk; }
HLE_FUNC(_ZN2nn4nifm20SubmitNetworkRequestEv, void, nifm_SubmitNetworkRequest, ()) {}
HLE_FUNC(_ZN2nn4nifm27SubmitNetworkRequestAndWaitEv, void, nifm_SubmitNetworkRequestAndWait, ()) {}
HLE_FUNC(_ZN2nn4nifm22IsNetworkRequestOnHoldEv, bool, nifm_IsNetworkRequestOnHold, ()) { return false; }
HLE_FUNC(_ZN2nn4nifm18IsNetworkAvailableEv, bool, nifm_IsNetworkAvailable, ()) { return false; }
HLE_FUNC(_ZN2nn4nifm26HandleNetworkRequestResultEv, uint32_t, nifm_HandleNetworkRequestResult, ()) {
    return kNifmNoNetwork;
}
HLE_FUNC(_ZN2nn4nifm19SetLocalNetworkModeEb, void, nifm_SetLocalNetworkMode, (bool)) {}
HLE_FUNC(_ZN2nn4nifm26GetCurrentPrimaryIpAddressEP7in_addr, uint32_t, nifm_GetCurrentPrimaryIpAddress, (void*)) {
    return kNifmNoNetwork;
}
HLE_FUNC(nnnifmGetCurrentProxySetting, uint32_t, nnnifmGetCurrentProxySetting, (void*)) { return kNifmNoNetwork; }

// ===== socket (BSD-style: return -1, errno via GetLastErrno) ==========================================
namespace {
thread_local int t_sock_errno = 0;
int SockFail() {
    t_sock_errno = kNetDown;
    return -1;
}
}  // namespace
HLE_FUNC(_ZN2nn6socket10InitializeERKNS0_6ConfigE, uint32_t, socket_InitializeConfig, (const void*)) { return kOk; }
HLE_FUNC(_ZN2nn6socket10InitializeEPvmmi, uint32_t, socket_Initialize, (void*, size_t, size_t, int)) { return kOk; }
HLE_FUNC(_ZN2nn6socket8FinalizeEv, uint32_t, socket_Finalize, ()) { return kOk; }
HLE_FUNC(_ZN2nn6socket12GetLastErrnoEv, int, socket_GetLastErrno, ()) { return t_sock_errno; }
HLE_ALIAS(nnsocketGetLastErrno, socket_GetLastErrno);
HLE_FUNC(nnsocketSetLastErrno, void, nnsocketSetLastErrno, (int e)) { t_sock_errno = e; }
HLE_FUNC(_ZN2nn6socket6SocketEiii, int, socket_Socket, (int, int, int)) { return SockFail(); }
HLE_ALIAS(nnsocketSocket, socket_Socket);
HLE_FUNC(_ZN2nn6socket5CloseEi, int, socket_Close, (int)) { return 0; }
HLE_ALIAS(nnsocketClose, socket_Close);
// Every other socket call fails the same way; argument lists don't matter for a constant failure.
HLE_FUNC(dsr_socket_fail, int, socket_GenericFail, ()) { return SockFail(); }
HLE_ALIAS(_ZN2nn6socket4BindEiPK8sockaddrj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket7ConnectEiPK8sockaddrj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket6ListenEii, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket6AcceptEiP8sockaddrPj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket4SendEiPKvmi, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket4RecvEiPvmi, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket6SendToEiPKvmiPK8sockaddrj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket8RecvFromEiPvmiP8sockaddrPj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket4PollEP6pollfdmi, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket6SelectEiP6fd_setS2_S2_P7timeval, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket10GetSockOptEiiiPvPj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket10SetSockOptEiiiPKvj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket11GetSockNameEiP8sockaddrPj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket11GetPeerNameEiP8sockaddrPj, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket5IoctlEijPvm, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket5FcntlEiiz, socket_GenericFail);
HLE_ALIAS(_ZN2nn6socket8ShutdownEii, socket_GenericFail);
HLE_ALIAS(nnsocketBind, socket_GenericFail);
HLE_ALIAS(nnsocketConnect, socket_GenericFail);
HLE_ALIAS(nnsocketSend, socket_GenericFail);
HLE_ALIAS(nnsocketRecv, socket_GenericFail);
HLE_ALIAS(nnsocketPoll, socket_GenericFail);
HLE_ALIAS(nnsocketGetSockOpt, socket_GenericFail);
HLE_ALIAS(nnsocketSetSockOpt, socket_GenericFail);
HLE_ALIAS(nnsocketGetSockName, socket_GenericFail);
HLE_ALIAS(nnsocketGetPeerName, socket_GenericFail);
HLE_ALIAS(nnsocketFcntl, socket_GenericFail);
HLE_ALIAS(nnsocketCancel, socket_GenericFail);
HLE_ALIAS(nnsocketRequestCancelHandle, socket_GenericFail);
HLE_FUNC(_ZN2nn6socket11GetAddrInfoEPKcS2_PK8addrinfoPPS3_, int, socket_GetAddrInfo,
         (const char*, const char*, const void*, void** res)) {
    if (res) *res = nullptr;
    return 2;  // EAI_AGAIN-style failure
}
HLE_FUNC(_ZN2nn6socket12FreeAddrInfoEP8addrinfo, void, socket_FreeAddrInfo, (void*)) {}
HLE_FUNC(nnsocketGetHostByNameWithOptions, void*, nnsocketGetHostByNameWithOptions, (const char*, const void*)) {
    t_sock_errno = kNetDown;
    return nullptr;
}
// Pure byte-order / text helpers work normally.
HLE_FUNC(_ZN2nn6socket9InetHtonsEt, uint16_t, socket_InetHtons, (uint16_t v)) { return htons(v); }
HLE_ALIAS(_ZN2nn6socket9InetNtohsEt, socket_InetHtons);
HLE_ALIAS(nnsocketInetHtons, socket_InetHtons);
HLE_ALIAS(nnsocketInetNtohs, socket_InetHtons);
HLE_FUNC(_ZN2nn6socket9InetHtonlEj, uint32_t, socket_InetHtonl, (uint32_t v)) { return htonl(v); }
HLE_ALIAS(_ZN2nn6socket9InetNtohlEj, socket_InetHtonl);
HLE_FUNC(_ZN2nn6socket8InetAtonEPKcP7in_addr, int, socket_InetAton, (const char* s, in_addr* a)) {
    return inet_aton(s, a);
}
HLE_FUNC(_ZN2nn6socket8InetNtoaE7in_addr, char*, socket_InetNtoa, (in_addr a)) { return inet_ntoa(a); }
HLE_FUNC(_ZN2nn6socket8InetNtopEiPKvPcj, const char*, socket_InetNtop, (int af, const void* src, char* dst, uint32_t n)) {
    return inet_ntop(af == 2 ? AF_INET : af, src, dst, n);  // Switch AF_INET = 2 like Linux
}
HLE_FUNC(_ZN2nn6socket8InetPtonEiPKcPv, int, socket_InetPton, (int af, const char* s, void* dst)) {
    return inet_pton(af, s, dst);
}

// ===== ssl ============================================================================================
HLE_FUNC(_ZN2nn3ssl10InitializeEv, uint32_t, ssl_Initialize, ()) { return kOk; }
HLE_FUNC(_ZN2nn3ssl8FinalizeEv, uint32_t, ssl_Finalize, ()) { return kOk; }
HLE_FUNC(_ZN2nn3ssl7Context6CreateENS1_10SslVersionE, uint32_t, ssl_ContextCreate, (void*, int)) {
    return kSslUnavailable;
}
HLE_FUNC(_ZN2nn3ssl7Context15ImportServerPkiEPmPKcjNS0_17CertificateFormatE, uint32_t, ssl_ImportServerPki,
         (void*, uint64_t*, const char*, uint32_t, int)) {
    return kSslUnavailable;
}
HLE_FUNC(nnsslInitialize, uint32_t, nnsslInitialize, ()) { return kOk; }
HLE_FUNC(nnsslFinalize, uint32_t, nnsslFinalize, ()) { return kOk; }
HLE_FUNC(dsr_ssl_fail, uint32_t, ssl_GenericFail, ()) { return kSslUnavailable; }
HLE_ALIAS(nnsslContextDestroy, ssl_GenericFail);
HLE_ALIAS(nnsslContextGetContextId, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionCreate, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionDestroy, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionSetOption, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionSetSocketDescriptor, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionSetHostName, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionSetVerifyOption, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionSetSessionCacheMode, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionSetIoMode, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionDoHandshakeWithCertBuffer, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionGetVerifyCertError, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionGetConnectionId, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionRead, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionWrite, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionPeek, ssl_GenericFail);
HLE_ALIAS(nnsslConnectionPending, ssl_GenericFail);

// ===== bcat (news / delivery cache) ===================================================================
HLE_FUNC(_ZN2nn4bcat10InitializeEv, void, bcat_Initialize, ()) {}
HLE_FUNC(dsr_bcat_fail, uint32_t, bcat_GenericFail, ()) { return kBcatUnavailable; }
HLE_FUNC(dsr_bcat_nop, void, bcat_Nop, ()) {}
HLE_ALIAS(_ZN2nn4bcat25MountDeliveryCacheStorageEv, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat27UnmountDeliveryCacheStorageEv, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat24RequestSyncDeliveryCacheEPNS0_21DeliveryCacheProgressE, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat31EnumerateDeliveryCacheDirectoryEPiPNS0_13DirectoryNameEi, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat17DeliveryCacheFileC1Ev, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat17DeliveryCacheFileD1Ev, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat17DeliveryCacheFile4OpenEPKc, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat17DeliveryCacheFile4OpenERKNS0_13DirectoryNameERKNS0_8FileNameE, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat17DeliveryCacheFile4ReadEPmlPvm, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat17DeliveryCacheFile5CloseEv, bcat_Nop);
HLE_FUNC(_ZN2nn4bcat17DeliveryCacheFile7GetSizeEv, int64_t, bcat_FileGetSize, (void*)) { return 0; }
HLE_ALIAS(_ZN2nn4bcat22DeliveryCacheDirectoryC1Ev, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat22DeliveryCacheDirectoryD1Ev, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat22DeliveryCacheDirectory4OpenERKNS0_13DirectoryNameE, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat22DeliveryCacheDirectory4ReadEPiPNS0_27DeliveryCacheDirectoryEntryEi, bcat_GenericFail);
HLE_ALIAS(_ZN2nn4bcat21DeliveryCacheProgressC1Ev, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat21DeliveryCacheProgressD1Ev, bcat_Nop);
HLE_ALIAS(_ZN2nn4bcat21DeliveryCacheProgress6UpdateEv, bcat_Nop);
HLE_FUNC(_ZNK2nn4bcat21DeliveryCacheProgress9GetStatusEv, int, bcat_ProgressGetStatus, (const void*)) {
    return 9;  // Done
}
HLE_FUNC(_ZNK2nn4bcat21DeliveryCacheProgress13GetWholeTotalEv, int64_t, bcat_ProgressGetWholeTotal, (const void*)) {
    return 0;
}
HLE_ALIAS(_ZNK2nn4bcat21DeliveryCacheProgress18GetWholeDownloadedEv, bcat_ProgressGetWholeTotal);

// ===== friends / nsd / ec ==============================================================================
HLE_FUNC(_ZN2nn7friends18GetBlockedUserListEPiPNS_7account23NetworkServiceAccountIdEii, uint32_t,
         friends_GetBlockedUserList, (int* count, void*, int, int)) {
    // The friends service is local on the console and works offline: an empty block list.
    if (count) *count = 0;
    return 0;
}
HLE_FUNC(_ZN2nn7friends18ShowUserDetailInfoERKNS_7account3UidERKNS1_23NetworkServiceAccountIdERKNS0_15InAppScreenNameESA_,
         uint32_t, friends_ShowUserDetailInfo, (const void*, const void*, const void*, const void*)) {
    return 0;  // the applet would show a profile page and return when closed
}
HLE_FUNC(_ZN2nn3nsd9ResolveExEPNS0_4FqdnERKS1_, uint32_t, nsd_ResolveEx, (char* out, const char* in)) {
    strncpy(out, in, 256);
    return kNsdUnavailable;
}
HLE_FUNC(_ZN2nn2ec30ShowShopApplicationInformationENS_13ApplicationIdE, void, ec_ShowShop, (uint64_t)) {
    LOGI("ec", "eShop page requested (ignored)");
}

// ===== nfp (amiibo): no reader ==========================================================================
HLE_FUNC(_ZN2nn3nfp10InitializeEv, void, nfp_Initialize, ()) {}
HLE_FUNC(_ZN2nn3nfp8FinalizeEv, void, nfp_Finalize, ()) {}
HLE_FUNC(_ZN2nn3nfp8GetStateEv, int, nfp_GetState, ()) { return 1; /* Initialized */ }
HLE_FUNC(_ZN2nn3nfp11ListDevicesEPNS0_12DeviceHandleEPii, uint32_t, nfp_ListDevices, (void*, int* count, int)) {
    *count = 0;
    return kNfpNeedRestart;
}
HLE_FUNC(_ZN2nn3nfp29AttachAvailabilityChangeEventEPNS_2os15SystemEventTypeE, uint32_t, nfp_AttachAvailabilityChange,
         (void* se)) {
    dsr::hle::os::InitSystemEvent(se, false, false);
    return kOk;
}
HLE_FUNC(_ZN2nn3nfp19AttachActivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE, uint32_t,
         nfp_AttachActivateEvent, (void* se, const void*)) {
    dsr::hle::os::InitSystemEvent(se, false, false);
    return kOk;
}
HLE_ALIAS(_ZN2nn3nfp21AttachDeactivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE, nfp_AttachActivateEvent);
HLE_FUNC(_ZN2nn3nfp14GetDeviceStateERKNS0_12DeviceHandleE, int, nfp_GetDeviceState, (const void*)) {
    return 0;  // Init
}
HLE_FUNC(dsr_nfp_fail, uint32_t, nfp_GenericFail, ()) { return kNfpNeedRestart; }
HLE_ALIAS(_ZN2nn3nfp14StartDetectionERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp13StopDetectionERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp5MountERKNS0_12DeviceHandleENS0_9ModelTypeENS0_11MountTargetE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp7UnmountERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp10GetTagInfoEPNS0_7TagInfoERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp13GetCommonInfoEPNS0_10CommonInfoERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp15GetRegisterInfoEPNS0_12RegisterInfoERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp9GetNpadIdEPjRKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp19OpenApplicationAreaERKNS0_12DeviceHandleEj, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp18GetApplicationAreaEPvPmRKNS0_12DeviceHandleEm, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp18SetApplicationAreaERKNS0_12DeviceHandleEPKvm, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp21CreateApplicationAreaERKNS0_12DeviceHandleERKNS0_25ApplicationAreaCreateInfoE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp5FlushERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp7RestoreERKNS0_12DeviceHandleE, nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp13StartRestorerEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE,
          nfp_GenericFail);
HLE_ALIAS(_ZN2nn3nfp19StartGameDataEraserEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE,
          nfp_GenericFail);
HLE_ALIAS(
    _ZN2nn3nfp29StartNicknameAndOwnerSettingsEPNS0_12DeviceHandleEPbPNS0_12RegisterInfoERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE,
    nfp_GenericFail);

// ===== ngc profanity filter: masks nothing =============================================================
HLE_FUNC(_ZN2nn3ngc15ProfanityFilterC1Ev, void, ngc_PF_ctor, (void*)) {}
HLE_FUNC(_ZN2nn3ngc15ProfanityFilterD1Ev, void, ngc_PF_dtor, (void*)) {}
HLE_FUNC(_ZN2nn3ngc15ProfanityFilter10InitializeEPvmb, uint32_t, ngc_PF_Initialize, (void*, void*, size_t, bool)) {
    return kOk;
}
HLE_FUNC(_ZN2nn3ngc15ProfanityFilter8FinalizeEv, uint32_t, ngc_PF_Finalize, (void*)) { return kOk; }
HLE_FUNC(_ZN2nn3ngc15ProfanityFilter11SetMaskModeENS1_8MaskModeE, void, ngc_PF_SetMaskMode, (void*, int)) {}
HLE_FUNC(_ZN2nn3ngc15ProfanityFilter24MaskProfanityWordsInTextEPiPtj, uint32_t, ngc_PF_Mask,
         (void*, int* count, uint16_t*, uint32_t)) {
    if (count) *count = 0;
    return kOk;
}

// ===== swkbd: keyboard applet is cancelled; inline keyboard is inert ===================================
HLE_FUNC(_ZN2nn5swkbd25GetRequiredWorkBufferSizeEi, size_t, swkbd_GetRequiredWorkBufferSize, (int)) { return 0x1000; }
HLE_FUNC(_ZN2nn5swkbd10MakePresetEPNS0_14KeyboardConfigENS0_6PresetE, void, swkbd_MakePreset, (void*, int)) {}
HLE_FUNC(_ZN2nn5swkbd14SetInitialTextEPNS0_15ShowKeyboardArgEPKDs, void, swkbd_SetInitialText,
         (void*, const char16_t*)) {}
HLE_FUNC(_ZN2nn5swkbd12ShowKeyboardEPNS0_6StringERKNS0_15ShowKeyboardArgE, uint32_t, swkbd_ShowKeyboard,
         (void*, const void*)) {
    LOGI("swkbd", "software keyboard requested; returning cancel");
    return kSwkbdCanceled;
}
HLE_FUNC(_ZN2nn5swkbd9InputText3SetEPKDs, void, swkbd_InputTextSet, (void*, const char16_t*)) {}
HLE_FUNC(dsr_swkbd_nop, void, swkbd_Nop, ()) {}

// ---- inline keyboard ------------------------------------------------------------------------------------
// The game drives nn::swkbd::InlineKeyboard: Initialize, register callbacks, Launch + Appear when it wants
// text, then Calc() every frame. We show an Android text dialog on Appear and, from Calc() (on the game's
// own thread), invoke the game's "decided enter" or "decided cancel" callback with the result.
// DecidedEnterArg as this game reads it: UTF-16 text inline at offset 0, length in characters at 0x3ec.
namespace {
struct InlineKeyboardState {
    std::mutex m;
    const void* on_finished_initialize = nullptr;
    const void* on_decided_enter = nullptr;
    const void* on_decided_cancel = nullptr;
    bool initialize_pending = false;
    bool waiting_for_text = false;
} g_ik;
alignas(16) uint8_t g_decided_enter_arg[0x400];
constexpr size_t kDecidedTextChars = 0x3ec / 2;
}  // namespace
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboardC1Ev, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboardD1Ev, swkbd_Nop);
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard8FinalizeEv, void, swkbd_IK_Finalize, (void*)) {
    std::lock_guard<std::mutex> lk(g_ik.m);
    g_ik.waiting_for_text = false;
    g_ik.initialize_pending = false;
}
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard11SetUtf8ModeEb, swkbd_Nop);  // the game uses UTF-16
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard12SetCursorPosEi, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard12SetInputTextERKNS0_9InputTextE, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard13SetWindowModeENS1_10WindowModeE, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard14SetKeytopScaleEf, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard18SetKeytopTranslateEff, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard22SetMovedCursorCallbackEPFvPKNS0_14MovedCursorArgEE, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard24SetChangedStringCallbackEPFvPKNS0_16ChangedStringArgEE, swkbd_Nop);
HLE_ALIAS(_ZN2nn5swkbd14InlineKeyboard27SetFinishedKeyboardCallbackEPFvvE, swkbd_Nop);
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard23SetDecidedEnterCallbackEPFvPKNS0_15DecidedEnterArgEE, void,
         swkbd_IK_SetDecidedEnter, (void*, const void* fn)) {
    std::lock_guard<std::mutex> lk(g_ik.m);
    g_ik.on_decided_enter = fn;
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard24SetDecidedCancelCallbackEPFvvE, void, swkbd_IK_SetDecidedCancel,
         (void*, const void* fn)) {
    std::lock_guard<std::mutex> lk(g_ik.m);
    g_ik.on_decided_cancel = fn;
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard29SetFinishedInitializeCallbackEPFvvE, void, swkbd_IK_SetFinishedInit,
         (void*, const void* fn)) {
    std::lock_guard<std::mutex> lk(g_ik.m);
    g_ik.on_finished_initialize = fn;
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard10InitializeEPv, bool, swkbd_IK_Initialize, (void*, void*)) { return true; }
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard6LaunchEv, bool, swkbd_IK_Launch, (void*)) {
    std::lock_guard<std::mutex> lk(g_ik.m);
    g_ik.initialize_pending = true;
    return true;
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard6AppearERKNS0_9AppearArgE, void, swkbd_IK_Appear, (void*, const void*)) {
    LOGI("swkbd", "keyboard requested: showing the text dialog");
    bool shown = dsr::platform::RequestText("Name", u"", 16);
    std::lock_guard<std::mutex> lk(g_ik.m);
    g_ik.waiting_for_text = true;
    if (!shown) dsr::platform::DeliverText(false, u"");  // no dialog: report a cancel
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard4CalcEv, uint32_t, swkbd_IK_Calc, (void*)) {
    const void* init_cb = nullptr;
    const void* enter_cb = nullptr;
    const void* cancel_cb = nullptr;
    bool ok = false;
    std::u16string text;
    {
        std::lock_guard<std::mutex> lk(g_ik.m);
        if (g_ik.initialize_pending) {
            g_ik.initialize_pending = false;
            init_cb = g_ik.on_finished_initialize;
        }
        if (g_ik.waiting_for_text && dsr::platform::TakeText(&ok, &text)) {
            g_ik.waiting_for_text = false;
            if (ok)
                enter_cb = g_ik.on_decided_enter;
            else
                cancel_cb = g_ik.on_decided_cancel;
        }
    }
    if (init_cb) dsr::arch::GuestCall<void>(init_cb);
    if (enter_cb) {
        memset(g_decided_enter_arg, 0, sizeof(g_decided_enter_arg));
        const size_t n = std::min(text.size(), kDecidedTextChars - 1);
        memcpy(g_decided_enter_arg, text.data(), n * 2);
        const int32_t len = int32_t(n);
        memcpy(g_decided_enter_arg + 0x3ec, &len, 4);
        LOGI("swkbd", "name entered (%d characters)", len);
        dsr::arch::GuestCall<void>(enter_cb, static_cast<const void*>(g_decided_enter_arg));
    }
    if (cancel_cb) dsr::arch::GuestCall<void>(cancel_cb);
    return 0;
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard25GetImageMemoryRequirementEPmS2_, void, swkbd_IK_GetImageMemoryRequirement,
         (size_t * size, size_t* align)) {
    *size = 0x1000;
    *align = 0x1000;
}
HLE_FUNC(_ZN2nn5swkbd14InlineKeyboard8GetImageEPvm, bool, swkbd_IK_GetImage, (void*, void*, size_t)) { return false; }
HLE_FUNC(_ZNK2nn5swkbd14InlineKeyboard13GetWindowSizeEPiS2_, void, swkbd_IK_GetWindowSize,
         (const void*, int* w, int* h)) {
    *w = 0;
    *h = 0;
}

// ===== account async network calls (always complete immediately with "unavailable") ===================
namespace {
constexpr uint32_t kAccountNsaUnavailable = Fail(124, 280);
}
HLE_FUNC(_ZN2nn7account12AsyncContextC1Ev, void, account_AsyncContext_ctor, (void*)) {}
HLE_FUNC(_ZN2nn7account12AsyncContextD1Ev, void, account_AsyncContext_dtor, (void*)) {}
HLE_FUNC(_ZN2nn7account12AsyncContext6CancelEv, uint32_t, account_AsyncContext_Cancel, (void*)) { return kOk; }
HLE_FUNC(_ZN2nn7account12AsyncContext7HasDoneEPb, uint32_t, account_AsyncContext_HasDone, (void*, bool* done)) {
    *done = true;
    return kOk;
}
HLE_FUNC(_ZN2nn7account12AsyncContext9GetResultEv, uint32_t, account_AsyncContext_GetResult, (void*)) {
    return kAccountNsaUnavailable;
}
HLE_FUNC(_ZN2nn7account12AsyncContext14GetSystemEventEPNS_2os11SystemEventE, uint32_t,
         account_AsyncContext_GetSystemEvent, (void*, void* se)) {
    dsr::hle::os::InitSystemEvent(se, false, true);  // already "done"
    return kOk;
}
HLE_FUNC(_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE, uint32_t,
         account_EnsureNsaIdTokenCacheAsync, (void*, const void*)) {
    return kOk;  // the context then reports failure via GetResult
}
HLE_FUNC(_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE, uint32_t,
         account_LoadNsaIdTokenCache, (size_t * size, char*, size_t, const void*)) {
    if (size) *size = 0;
    return kAccountNsaUnavailable;
}

// ===== friends play history / nifm proxy / socket errors ==============================================
HLE_FUNC(_ZN2nn7friends29GetPlayHistoryRegistrationKeyEPNS0_26PlayHistoryRegistrationKeyERKNS_7account3UidEb, uint32_t,
         friends_GetPlayHistoryRegistrationKey, (void* key, const void*, bool)) {
    // A 64-byte opaque key generated locally on the console; any stable value works.
    if (key) {
        memset(key, 0, 0x40);
        memcpy(key, "DSRNATIVE-PLAYHISTORY", 21);
    }
    return 0;
}
HLE_FUNC(_ZN2nn7friends14AddPlayHistoryERKNS_7account3UidERKNS0_26PlayHistoryRegistrationKeyERKNS0_15InAppScreenNameESA_,
         uint32_t, friends_AddPlayHistory, (const void*, const void*, const void*, const void*)) {
    return 0;  // recorded locally on the console; nothing to keep here
}
HLE_FUNC(_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE, uint32_t, nifm_GetCurrentProxySetting, (void*)) {
    return kNifmNoNetwork;
}
HLE_FUNC(_ZN2nn6socket11GAIStrErrorEi, const char*, socket_GAIStrError, (int)) { return "network unavailable"; }
