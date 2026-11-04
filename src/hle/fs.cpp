// nn::fs — mount points are mapped onto host directories:
//   MountRom(name)      -> RuntimeConfig::game_dir (the extracted romfs: dvdbnd*.bdt, movww/, ...)
//   MountSaveData(name) -> RuntimeConfig::save_dir
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/log.h"
#include "hle/registry.h"
#include "runtime/runtime.h"

namespace {
constexpr const char* kTag = "fs";

// nn::fs results (module 2)
constexpr uint32_t MakeResult(uint32_t desc) { return 2u | (desc << 9); }
constexpr uint32_t kOk = 0;
constexpr uint32_t kPathNotFound = MakeResult(1);
constexpr uint32_t kPathAlreadyExists = MakeResult(2);
constexpr uint32_t kTargetLocked = MakeResult(7);
constexpr uint32_t kDirectoryNotEmpty = MakeResult(8);
constexpr uint32_t kUsableSpaceNotEnough = MakeResult(30);
constexpr uint32_t kMountNameAlreadyExists = MakeResult(60);
constexpr uint32_t kOutOfRange = MakeResult(3005);
constexpr uint32_t kInvalidPath = MakeResult(6001);
constexpr uint32_t kFileExtensionWithoutOpenModeAllowAppend = MakeResult(6201);
constexpr uint32_t kUnexpected = MakeResult(5000);

enum OpenMode { kRead = 1, kWrite = 2, kAllowAppend = 4 };

std::mutex g_mounts_mutex;
std::unordered_map<std::string, std::string> g_mounts;  // name -> host dir (no trailing slash)

struct HostFile {
    int fd;
    int mode;
    std::string path;
};
struct HostDir {
    std::vector<std::pair<std::string, struct stat>> entries;
    size_t pos = 0;
};

struct DirectoryEntry {  // nn::fs::DirectoryEntry, 0x310 bytes
    char name[0x301];
    uint8_t reserved0[3];
    int8_t type;  // 0 = directory, 1 = file
    uint8_t reserved1[3];
    int64_t size;
};
static_assert(sizeof(DirectoryEntry) == 0x310, "DirectoryEntry layout");

uint32_t FromErrno(int e) {
    switch (e) {
        case ENOENT:
        case ENOTDIR: return kPathNotFound;
        case EEXIST: return kPathAlreadyExists;
        case ENOTEMPTY: return kDirectoryNotEmpty;
        case ENOSPC: return kUsableSpaceNotEnough;
        case EBUSY: return kTargetLocked;
        default: return kUnexpected;
    }
}

// "name:/a/b" -> host path. Returns false for unknown mounts.
bool Resolve(const char* guest, std::string* out) {
    if (!guest) return false;
    const char* colon = strchr(guest, ':');
    if (!colon) return false;
    std::string mount(guest, colon - guest);
    const char* rest = colon + 1;
    while (*rest == '/' || *rest == '\\') rest++;
    std::lock_guard<std::mutex> lk(g_mounts_mutex);
    auto it = g_mounts.find(mount);
    if (it == g_mounts.end()) return false;
    *out = it->second + "/" + rest;
    for (char& c : *out)
        if (c == '\\') c = '/';
    return true;
}

#define RESOLVE_OR_FAIL(guest, host)                                      \
    std::string host;                                                     \
    if (!Resolve(guest, &host)) {                                         \
        LOGW(kTag, "%s: unresolvable path '%s'", __func__, guest ? guest : "(null)"); \
        return kPathNotFound;                                             \
    }

uint32_t Mount(const char* name, const std::string& dir) {
    mkdir(dir.c_str(), 0775);
    std::lock_guard<std::mutex> lk(g_mounts_mutex);
    if (!g_mounts.emplace(name, dir).second) return kMountNameAlreadyExists;
    LOGI(kTag, "mounted %s: -> %s", name, dir.c_str());
    return kOk;
}

int RemoveTree(const std::string& path) {
    DIR* d = opendir(path.c_str());
    if (!d) return -1;
    while (dirent* e = readdir(d)) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        std::string child = path + "/" + e->d_name;
        struct stat st;
        if (lstat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            RemoveTree(child);
        else
            unlink(child.c_str());
    }
    closedir(d);
    return rmdir(path.c_str());
}
}  // namespace

// ---- mounting ------------------------------------------------------------------------------------
HLE_FUNC(_ZN2nn2fs22QueryMountRomCacheSizeEPm, uint32_t, fs_QueryMountRomCacheSize, (size_t * out)) {
    *out = 0x1000;  // no cache needed: reads go straight to the host filesystem
    return kOk;
}
HLE_FUNC(_ZN2nn2fs8MountRomEPKcPvm, uint32_t, fs_MountRom, (const char* name, void*, size_t)) {
    return Mount(name, dsr::GetRuntime().config.game_dir);
}
HLE_FUNC(_ZN2nn2fs14EnsureSaveDataERKNS_7account3UidE, uint32_t, fs_EnsureSaveData, (const void*)) {
    mkdir(dsr::GetRuntime().config.save_dir.c_str(), 0775);
    return kOk;
}
HLE_FUNC(_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE, uint32_t, fs_MountSaveData, (const char* name, const void*)) {
    return Mount(name, dsr::GetRuntime().config.save_dir);
}
HLE_FUNC(_ZN2nn2fs7UnmountEPKc, void, fs_Unmount, (const char* name)) {
    std::lock_guard<std::mutex> lk(g_mounts_mutex);
    g_mounts.erase(name);
}
HLE_FUNC(_ZN2nn2fs14CommitSaveDataEPKc, uint32_t, fs_CommitSaveData, (const char*)) {
    sync();
    return kOk;
}
HLE_FUNC(_ZN2nn2fs12SetAllocatorEPFPvmEPFvS1_mE, void, fs_SetAllocator, (const void*, const void*)) {}

// ---- files ----------------------------------------------------------------------------------------
HLE_FUNC(_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci, uint32_t, fs_OpenFile, (void** out, const char* path, int mode)) {
    RESOLVE_OR_FAIL(path, host);
    int flags = (mode & kWrite) ? ((mode & kRead) ? O_RDWR : O_WRONLY) : O_RDONLY;
    int fd = open(host.c_str(), flags | O_CLOEXEC);
    if (fd < 0) {
        LOGD(kTag, "OpenFile('%s') failed: %s", path, strerror(errno));
        return FromErrno(errno);
    }
    *out = new HostFile{fd, mode, host};
    LOGD(kTag, "OpenFile('%s', %d) -> fd %d", path, mode, fd);
    return kOk;
}
HLE_FUNC(_ZN2nn2fs9CloseFileENS0_10FileHandleE, void, fs_CloseFile, (HostFile * f)) {
    if (!f) return;
    close(f->fd);
    delete f;
}
static uint32_t ReadImpl(size_t* out, HostFile* f, int64_t off, void* buf, size_t size) {
    size_t done = 0;
    while (done < size) {
        ssize_t r = pread(f->fd, static_cast<char*>(buf) + done, size - done, off + int64_t(done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return FromErrno(errno);
        }
        if (r == 0) break;
        done += size_t(r);
    }
    if (out) *out = done;
    return kOk;
}
HLE_FUNC(_ZN2nn2fs8ReadFileENS0_10FileHandleElPvm, uint32_t, fs_ReadFile,
         (HostFile * f, int64_t off, void* buf, size_t size)) {
    size_t n = 0;
    uint32_t r = ReadImpl(&n, f, off, buf, size);
    if (r == kOk && n != size) return kOutOfRange;
    return r;
}
HLE_FUNC(_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm, uint32_t, fs_ReadFileOut,
         (size_t * out, HostFile* f, int64_t off, void* buf, size_t size)) {
    return ReadImpl(out, f, off, buf, size);
}
HLE_FUNC(_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE, uint32_t, fs_WriteFile,
         (HostFile * f, int64_t off, const void* buf, size_t size, const int* option)) {
    if (!(f->mode & kAllowAppend)) {
        struct stat st;
        if (fstat(f->fd, &st) == 0 && off + int64_t(size) > st.st_size) return kFileExtensionWithoutOpenModeAllowAppend;
    }
    size_t done = 0;
    while (done < size) {
        ssize_t r = pwrite(f->fd, static_cast<const char*>(buf) + done, size - done, off + int64_t(done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return FromErrno(errno);
        }
        done += size_t(r);
    }
    if (option && (*option & 1)) fsync(f->fd);  // WriteOptionFlag_Flush
    return kOk;
}
HLE_FUNC(_ZN2nn2fs9FlushFileENS0_10FileHandleE, uint32_t, fs_FlushFile, (HostFile * f)) {
    fsync(f->fd);
    return kOk;
}
HLE_FUNC(_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE, uint32_t, fs_GetFileSize, (int64_t * out, HostFile* f)) {
    struct stat st;
    if (fstat(f->fd, &st) != 0) return FromErrno(errno);
    *out = st.st_size;
    return kOk;
}
HLE_FUNC(_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl, uint32_t, fs_SetFileSize, (HostFile * f, int64_t size)) {
    return ftruncate(f->fd, size) == 0 ? kOk : FromErrno(errno);
}
HLE_FUNC(_ZN2nn2fs10CreateFileEPKcl, uint32_t, fs_CreateFile, (const char* path, int64_t size)) {
    RESOLVE_OR_FAIL(path, host);
    int fd = open(host.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0664);
    if (fd < 0) return FromErrno(errno);
    uint32_t r = ftruncate(fd, size) == 0 ? kOk : FromErrno(errno);
    close(fd);
    return r;
}
HLE_FUNC(_ZN2nn2fs10DeleteFileEPKc, uint32_t, fs_DeleteFile, (const char* path)) {
    RESOLVE_OR_FAIL(path, host);
    return unlink(host.c_str()) == 0 ? kOk : FromErrno(errno);
}
HLE_FUNC(_ZN2nn2fs10RenameFileEPKcS2_, uint32_t, fs_RenameFile, (const char* from, const char* to)) {
    RESOLVE_OR_FAIL(from, a);
    RESOLVE_OR_FAIL(to, b);
    struct stat st;
    if (stat(b.c_str(), &st) == 0) return kPathAlreadyExists;
    return rename(a.c_str(), b.c_str()) == 0 ? kOk : FromErrno(errno);
}

// ---- directories ------------------------------------------------------------------------------------
HLE_FUNC(_ZN2nn2fs15CreateDirectoryEPKc, uint32_t, fs_CreateDirectory, (const char* path)) {
    RESOLVE_OR_FAIL(path, host);
    return mkdir(host.c_str(), 0775) == 0 ? kOk : FromErrno(errno);
}
HLE_FUNC(_ZN2nn2fs15DeleteDirectoryEPKc, uint32_t, fs_DeleteDirectory, (const char* path)) {
    RESOLVE_OR_FAIL(path, host);
    return rmdir(host.c_str()) == 0 ? kOk : FromErrno(errno);
}
HLE_FUNC(_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc, uint32_t, fs_DeleteDirectoryRecursively, (const char* path)) {
    RESOLVE_OR_FAIL(path, host);
    return RemoveTree(host) == 0 ? kOk : FromErrno(errno);
}
HLE_FUNC(_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc, uint32_t, fs_GetEntryType,
         (int* out, const char* path)) {
    RESOLVE_OR_FAIL(path, host);
    struct stat st;
    if (stat(host.c_str(), &st) != 0) return FromErrno(errno);
    *out = S_ISDIR(st.st_mode) ? 0 : 1;
    return kOk;
}
HLE_FUNC(_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci, uint32_t, fs_OpenDirectory,
         (void** out, const char* path, int mode)) {
    RESOLVE_OR_FAIL(path, host);
    DIR* d = opendir(host.c_str());
    if (!d) return FromErrno(errno);
    auto* hd = new HostDir;
    while (dirent* e = readdir(d)) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (stat((host + "/" + e->d_name).c_str(), &st) != 0) continue;
        bool is_dir = S_ISDIR(st.st_mode);
        if ((is_dir && (mode & 1)) || (!is_dir && (mode & 2))) hd->entries.emplace_back(e->d_name, st);
    }
    closedir(d);
    *out = hd;
    return kOk;
}
HLE_FUNC(_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE, void, fs_CloseDirectory, (HostDir * d)) { delete d; }
HLE_FUNC(_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl, uint32_t, fs_ReadDirectory,
         (int64_t * out_count, DirectoryEntry* entries, HostDir* d, int64_t max)) {
    int64_t n = 0;
    while (n < max && d->pos < d->entries.size()) {
        const auto& [name, st] = d->entries[d->pos++];
        DirectoryEntry& e = entries[n++];
        memset(&e, 0, sizeof(e));
        strncpy(e.name, name.c_str(), sizeof(e.name) - 1);
        e.type = S_ISDIR(st.st_mode) ? 0 : 1;
        e.size = S_ISDIR(st.st_mode) ? 0 : st.st_size;
    }
    *out_count = n;
    return kOk;
}
HLE_FUNC(_ZN2nn2fs22GetDirectoryEntryCountEPlNS0_15DirectoryHandleE, uint32_t, fs_GetDirectoryEntryCount,
         (int64_t * out, HostDir* d)) {
    *out = int64_t(d->entries.size());
    return kOk;
}

// Keep unused-constant warnings quiet for results we may need later.
[[maybe_unused]] static const uint32_t kUnusedResults[] = {kInvalidPath};
