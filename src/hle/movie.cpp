// movie:: (multimedia.nss / subsdk0) — video playback for the title-screen background, the intro and
// the ending.
//
// subsdk0 is a port of Android's Stagefright (NuPlayer/ACodec), and the movie:: API mirrors Android's
// MediaExtractor/MediaCodec: MediaFormat keys ("mime", "width", "sample-rate", ...), "end of stream"
// status, asynchronous decoders that report available buffers through events. On Android it is
// therefore implemented directly on the NDK's AMediaExtractor/AMediaCodec, using the device's hardware
// decoders. Other hosts get an inert "no tracks" extractor.
//
// Object layout (from the game's code): movie::Extractor and movie::Decoder are 16 bytes — a vptr and
// an implementation pointer at +8. The game calls most methods through the vtable, so the constructors
// install host vtables laid out like subsdk0's (_ZTVN5movie9ExtractorE / _ZTVN5movie7DecoderE).
//
// Status codes: 0 = OK, 0x1000 = generic failure, 0x2000 = end of stream (checked by the extractor
// thread). DecoderEvents = {Event* input_available, output_available, format_changed, error}.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hle/nvn/backend.h"
#include "common/log.h"
#include "hle/fs.h"
#include "hle/os/heap.h"
#include "hle/os/os.h"
#include "hle/registry.h"
#include "loader/dynamic_thunks.h"
#include "runtime/runtime.h"

#if defined(__ANDROID__)
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#endif

namespace {
constexpr const char* kTag = "movie";
constexpr int kOk = 0;
constexpr int kFail = 0x1000;
constexpr int kEndOfStream = 0x2000;

void Trace(const char* what) {
    static std::mutex m;
    static std::unordered_map<const char*, int> counts;
    std::lock_guard<std::mutex> lk(m);
    if (++counts[what] <= 3) LOGI(kTag, "%s", what);
}

void FreeGuestObject(void* p) {
    if (!p) return;
    if (dsr::hle::os::GuestHeapContains(p))
        dsr::GuestFree(p);
    else
        free(p);
}

template <typename T>
T*& ImplOf(void* self) {
    return reinterpret_cast<T**>(self)[1];
}

// ---- MediaData: key/value metadata, kept host-side ---------------------------------------------------
struct MediaData {
    std::map<std::string, std::string> strings;
    std::map<std::string, int64_t> ints;
#if defined(__ANDROID__)
    std::shared_ptr<AMediaFormat> format;  // original format (carries codec-specific data, csd-0/1)
#endif
};
std::mutex g_md_mutex;
std::unordered_map<const void*, MediaData> g_media_data;

[[maybe_unused]] MediaData& MD(const void* self) {
    std::lock_guard<std::mutex> lk(g_md_mutex);
    return g_media_data[self];
}

#if defined(__ANDROID__)
void FillMediaData(MediaData& md, AMediaFormat* fmt, bool take_ownership) {
    md.strings.clear();
    md.ints.clear();
    const char* mime = nullptr;
    if (AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime) && mime) md.strings["mime"] = mime;
    const char* int_keys[] = {"width", "height", "sample-rate", "channel-count", "max-input-size", "color-format",
                              "stride", "slice-height", "crop-left", "crop-top", "crop-right", "crop-bottom"};
    for (const char* k : int_keys) {
        int32_t v;
        if (AMediaFormat_getInt32(fmt, k, &v)) md.ints[k] = v;
    }
    int64_t d;
    if (AMediaFormat_getInt64(fmt, AMEDIAFORMAT_KEY_DURATION, &d)) md.ints["durationUs"] = d;
    if (take_ownership) md.format.reset(fmt, AMediaFormat_delete);
}
#endif

// ---- Buffer: kept host-side ---------------------------------------------------------------------------
struct BufferState {
    uint8_t* data = nullptr;
    size_t capacity = 0;
    int offset = 0;
    int size = 0;
    bool owned = false;
    bool from_game = false;  // memory supplied by the game (SetDataAndCapacity): decoders write into it
};
std::mutex g_buf_mutex;
std::unordered_map<const void*, BufferState> g_buffers;

BufferState& Buf(const void* self) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    return g_buffers[self];
}
void SetBuffer(const void* self, uint8_t* data, size_t capacity, int offset, int size) {
    BufferState& b = Buf(self);
    if (b.owned) free(b.data);
    b = {data, capacity, offset, size, false};
}

// ===== Extractor ==========================================================================================
struct Extractor {
    int fd = -1;
#if defined(__ANDROID__)
    AMediaExtractor* ex = nullptr;
#endif
    ~Extractor() {
#if defined(__ANDROID__)
        if (ex) AMediaExtractor_delete(ex);
#endif
        if (fd >= 0) close(fd);
    }
};
Extractor* Ex(void* self) { return ImplOf<Extractor>(self); }

void ExDtor(void* self) {
    Trace("Extractor::~Extractor");
    delete Ex(self);
    ImplOf<Extractor>(self) = nullptr;
}
void ExDtorDelete(void* self) {
    ExDtor(self);
    FreeGuestObject(self);
}
int ExSetDataSourcePath(void* self, const char* path) {
    std::string host;
    if (!path || !dsr::hle::fs::ResolveGuestPath(path, &host)) {
        LOGW(kTag, "Extractor::SetDataSource: cannot resolve %s", path ? path : "(null)");
        return kFail;
    }
    Extractor* e = Ex(self);
    e->fd = open(host.c_str(), O_RDONLY | O_CLOEXEC);
    if (e->fd < 0) {
        LOGW(kTag, "Extractor::SetDataSource: cannot open %s", host.c_str());
        return kFail;
    }
#if defined(__ANDROID__)
    struct stat st;
    fstat(e->fd, &st);
    e->ex = AMediaExtractor_new();
    media_status_t r = AMediaExtractor_setDataSourceFd(e->ex, e->fd, 0, st.st_size);
    LOGI(kTag, "Extractor::SetDataSource(%s): %s, %zu tracks", path, r == AMEDIA_OK ? "ok" : "FAILED",
         r == AMEDIA_OK ? AMediaExtractor_getTrackCount(e->ex) : size_t(0));
    if (r == AMEDIA_OK) dsr::hle::nvn::GetBackend().RequestCapture(8);  // diagnostics: what a video frame shows
    return r == AMEDIA_OK ? kOk : kFail;
#else
    LOGI(kTag, "Extractor::SetDataSource(%s): no media backend on this host (0 tracks)", path);
    return kOk;
#endif
}
int ExSetDataSourceMem(void*, const void*, int) {
    LOGW(kTag, "Extractor::SetDataSource(memory) not supported");
    return kFail;
}
int ExGetTrackCount(void* self, int* count) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    *count = e->ex ? int(AMediaExtractor_getTrackCount(e->ex)) : 0;
#else
    *count = 0;
#endif
    return kOk;
}
int ExGetTrackConfiguration(void* self, int index, void* md) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    if (!e->ex) return kFail;
    AMediaFormat* fmt = AMediaExtractor_getTrackFormat(e->ex, size_t(index));
    if (!fmt) return kFail;
    MediaData& m = MD(md);
    FillMediaData(m, fmt, true);
    LOGI(kTag, "track %d: %s %s", index, m.strings["mime"].c_str(), AMediaFormat_toString(fmt));
    return kOk;
#else
    return kFail;
#endif
}
int ExSelectTrack(void* self, int index) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    return e->ex && AMediaExtractor_selectTrack(e->ex, size_t(index)) == AMEDIA_OK ? kOk : kFail;
#else
    return kFail;
#endif
}
int ExUnselectTrack(void* self, int index) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    if (e->ex) AMediaExtractor_unselectTrack(e->ex, size_t(index));
#endif
    return kOk;
}
int ExSeekTo(void* self, int64_t us) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    if (e->ex) AMediaExtractor_seekTo(e->ex, us, AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC);
#endif
    return kOk;
}
int ExGetTrackIndexForAvailableData(void* self, size_t* index) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    int t = e->ex ? AMediaExtractor_getSampleTrackIndex(e->ex) : -1;
    if (t < 0) return kEndOfStream;
    *index = size_t(t);
    return kOk;
#else
    return kEndOfStream;
#endif
}
int ExRead(void* self, void* buffer) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    BufferState& b = Buf(buffer);
    if (!e->ex) return kFail;
    ssize_t n = AMediaExtractor_readSampleData(e->ex, b.data, b.capacity);
    if (n < 0) return kEndOfStream;
    b.offset = 0;
    b.size = int(n);
    return kOk;
#else
    return kEndOfStream;
#endif
}
int ExAdvance(void* self) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    return e->ex && AMediaExtractor_advance(e->ex) ? kOk : kEndOfStream;
#else
    return kEndOfStream;
#endif
}
int ExGetSampleTime(void* self, int64_t* t) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    int64_t v = e->ex ? AMediaExtractor_getSampleTime(e->ex) : -1;
    if (v < 0) return kEndOfStream;
    *t = v;
    return kOk;
#else
    *t = 0;
    return kEndOfStream;
#endif
}
int ExGetSampleMetaData(void* self, void* md) {
#if defined(__ANDROID__)
    Extractor* e = Ex(self);
    if (!e->ex) return kFail;
    MediaData& m = MD(md);
    uint32_t flags = AMediaExtractor_getSampleFlags(e->ex);
    m.ints["is-sync-frame"] = (flags & AMEDIAEXTRACTOR_SAMPLE_FLAG_SYNC) ? 1 : 0;
    m.ints["timeUs"] = AMediaExtractor_getSampleTime(e->ex);
    return kOk;
#else
    return kFail;
#endif
}
int ExGetBufferedRange(void*, int64_t* a, int64_t* b) {
    *a = 0;
    *b = INT64_MAX / 2;  // local file: everything is "buffered"
    return kOk;
}
int ExSetDataSourceReader(void*, void*, const char*) {
    LOGW(kTag, "Extractor::SetDataSource(reader) not supported");
    return kFail;
}

// ===== Decoder ============================================================================================
struct DecoderEvents {
    void* input_available;
    void* output_available;
    void* format_changed;
    void* error;
};

struct OutputItem {
    int index;
    int32_t offset, size;
    int64_t time_us;
    uint32_t flags;
};

struct Decoder {
    int type = 0, output_format = 0, mode = 0;
    DecoderEvents events{};
    std::mutex m;
    std::deque<int> inputs;
    std::deque<OutputItem> outputs;
    std::unordered_map<int, OutputItem> held;  // output buffers handed to the game
    uint64_t produced = 0, acquired = 0, released = 0, polls_empty = 0;  // statistics (logged)
    int64_t last_pts = 0;
    // Decoded picture geometry (from the codec's output format).
    int width = 0, height = 0, stride = 0, slice_height = 0, crop_left = 0, crop_top = 0;
    bool logged_convert = false;
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
#if defined(__ANDROID__)
    AMediaCodec* codec = nullptr;
#endif
    ~Decoder() {
#if defined(__ANDROID__)
        if (codec) {
            AMediaCodec_stop(codec);
            AMediaCodec_delete(codec);
        }
#endif
    }
};
Decoder* Dec(void* self) { return ImplOf<Decoder>(self); }

#if defined(__ANDROID__)
void OnInput(AMediaCodec*, void* ud, int32_t index) {
    auto* d = static_cast<Decoder*>(ud);
    {
        std::lock_guard<std::mutex> lk(d->m);
        d->inputs.push_back(index);
    }
    dsr::hle::os::SignalGuestEvent(d->events.input_available);
}
void OnOutput(AMediaCodec*, void* ud, int32_t index, AMediaCodecBufferInfo* info) {
    auto* d = static_cast<Decoder*>(ud);
    {
        std::lock_guard<std::mutex> lk(d->m);
        d->outputs.push_back({index, info->offset, info->size, info->presentationTimeUs, info->flags});
        d->produced++;
    }
    dsr::hle::os::SignalGuestEvent(d->events.output_available);
}
void OnFormat(AMediaCodec*, void* ud, AMediaFormat* fmt) {
    auto* d = static_cast<Decoder*>(ud);
    LOGI(kTag, "decoder output format: %s", AMediaFormat_toString(fmt));
    {
        std::lock_guard<std::mutex> lk(d->m);
        int32_t v;
        if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, &v)) d->width = v;
        if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, &v)) d->height = v;
        d->stride = AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_STRIDE, &v) ? v : d->width;
        d->slice_height = AMediaFormat_getInt32(fmt, "slice-height", &v) ? v : d->height;
        int32_t l, t, r, b;
        if (AMediaFormat_getRect(fmt, "crop", &l, &t, &r, &b)) {
            d->crop_left = l;
            d->crop_top = t;
            d->width = r - l + 1;
            d->height = b - t + 1;
        }
        if (d->stride < d->width) d->stride = d->width;
        if (d->slice_height < d->height) d->slice_height = d->height;
    }
    dsr::hle::os::SignalGuestEvent(d->events.format_changed);
}
void OnError(AMediaCodec*, void* ud, media_status_t err, int32_t code, const char* detail) {
    auto* d = static_cast<Decoder*>(ud);
    LOGE(kTag, "decoder error %d (%d): %s", int(err), code, detail ? detail : "");
    dsr::hle::os::SignalGuestEvent(d->events.error);
}
#endif

// NV12 (BT.709, limited range) -> RGBA8, two pixels per chroma sample, table driven.
struct YuvTables {
    int y[256], rv[256], gu[256], gv[256], bu[256];
    YuvTables() {
        for (int i = 0; i < 256; i++) {
            y[i] = 298 * (i - 16) + 128;
            rv[i] = 459 * (i - 128);   // 1.793
            gu[i] = -55 * (i - 128);   // 0.213
            gv[i] = -136 * (i - 128);  // 0.533
            bu[i] = 541 * (i - 128);   // 2.112
        }
    }
};
inline uint8_t Clamp8(int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); }
// Rows [y0, y1). The decoder's output buffer is uncached memory on some devices (slow to read byte by
// byte), so each source row is first copied (wide loads) into a cached row buffer.
void ConvertNv12Rows(const uint8_t* y_plane, const uint8_t* uv_plane, int stride, int cl, int ct, int w, int y0,
                     int y1, uint8_t* dst, size_t pitch) {
    static const YuvTables T;
    std::vector<uint8_t> yrow(size_t(w) + 2), uvrow(size_t(w) + 4);
    int uv_loaded = -1;
    for (int yy = y0; yy < y1; yy++) {
        memcpy(yrow.data(), y_plane + size_t(yy + ct) * size_t(stride) + cl, size_t(w));
        const int uvy = (yy + ct) / 2;
        if (uvy != uv_loaded) {
            memcpy(uvrow.data(), uv_plane + size_t(uvy) * size_t(stride) + (cl & ~1), size_t((w + 1) & ~1));
            uv_loaded = uvy;
        }
        const uint8_t* yr = yrow.data();
        const uint8_t* uvr = uvrow.data();
        uint32_t* out = reinterpret_cast<uint32_t*>(dst + size_t(yy) * pitch);
        int xx = 0;
        for (; xx + 1 < w; xx += 2) {
            const int u = uvr[xx], v = uvr[xx + 1];
            const int dr = T.rv[v], dg = T.gu[u] + T.gv[v], db = T.bu[u];
            const int c0 = T.y[yr[xx]], c1 = T.y[yr[xx + 1]];
            out[xx] = uint32_t(Clamp8((c0 + dr) >> 8)) | uint32_t(Clamp8((c0 + dg) >> 8)) << 8 |
                      uint32_t(Clamp8((c0 + db) >> 8)) << 16 | 0xff000000u;
            out[xx + 1] = uint32_t(Clamp8((c1 + dr) >> 8)) | uint32_t(Clamp8((c1 + dg) >> 8)) << 8 |
                          uint32_t(Clamp8((c1 + db) >> 8)) << 16 | 0xff000000u;
        }
        if (xx < w) {
            const int u = uvr[xx], v = uvr[xx + 1], c = T.y[yr[xx]];
            out[xx] = uint32_t(Clamp8((c + T.rv[v]) >> 8)) | uint32_t(Clamp8((c + T.gu[u] + T.gv[v]) >> 8)) << 8 |
                      uint32_t(Clamp8((c + T.bu[u]) >> 8)) << 16 | 0xff000000u;
        }
    }
}
// The whole picture, split over 4 threads (a 1080p frame is ~2 M pixels, 30 times a second).
void ConvertNv12(const uint8_t* y_plane, int stride, int slice, int cl, int ct, int w, int h, uint8_t* dst,
                 size_t pitch) {
    const uint8_t* uv_plane = y_plane + size_t(stride) * size_t(slice);
    constexpr int kParts = 4;
    const int band = ((h + kParts - 1) / kParts + 1) & ~1;  // even, so chroma rows are not split
    std::thread workers[kParts - 1];
    for (int i = 1; i < kParts; i++) {
        const int y0 = std::min(h, band * i), y1 = std::min(h, band * (i + 1));
        if (y0 < y1)
            workers[i - 1] = std::thread(ConvertNv12Rows, y_plane, uv_plane, stride, cl, ct, w, y0, y1, dst, pitch);
    }
    ConvertNv12Rows(y_plane, uv_plane, stride, cl, ct, w, 0, std::min(h, band), dst, pitch);
    for (auto& t : workers)
        if (t.joinable()) t.join();
}
// Average conversion time, logged every 120 frames.
void RecordConvertTime(std::chrono::steady_clock::duration d) {
    static std::mutex m;
    static double total_ms = 0, max_ms = 0;
    static int n = 0;
    std::lock_guard<std::mutex> lk(m);
    const double ms = std::chrono::duration<double, std::milli>(d).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);
    if (++n == 120) {
        LOGI(kTag, "video frame conversion: %.1f ms average, %.1f ms max (120 frames)", total_ms / n, max_ms);
        total_ms = max_ms = 0;
        n = 0;
    }
}

void DecDtor(void* self) {
    Trace("Decoder::~Decoder");
    delete Dec(self);
    ImplOf<Decoder>(self) = nullptr;
}
void DecDtorDelete(void* self) {
    DecDtor(self);
    FreeGuestObject(self);
}
int DecConfigure(void* self, void* md) {
    Decoder* d = Dec(self);
#if defined(__ANDROID__)
    MediaData& m = MD(md);
    const std::string mime = m.strings["mime"];
    if (mime.empty() || !m.format) {
        LOGW(kTag, "Decoder::Configure: no format");
        return kFail;
    }
    d->codec = AMediaCodec_createDecoderByType(mime.c_str());
    if (!d->codec) {
        LOGE(kTag, "no decoder for %s", mime.c_str());
        return kFail;
    }
    AMediaCodecOnAsyncNotifyCallback cb{OnInput, OnOutput, OnFormat, OnError};
    AMediaCodec_setAsyncNotifyCallback(d->codec, cb, d);
    if (mime.rfind("video/", 0) == 0) AMediaFormat_setInt32(m.format.get(), "color-format", 21);  // NV12
    media_status_t r = AMediaCodec_configure(d->codec, m.format.get(), nullptr, nullptr, 0);
    LOGI(kTag, "Decoder::Configure(%s): %s", mime.c_str(), r == AMEDIA_OK ? "ok" : "FAILED");
    return r == AMEDIA_OK ? kOk : kFail;
#else
    (void)d;
    return kFail;
#endif
}
int DecRegisterClientEvents(void* self, void* events) {
    if (events) Dec(self)->events = *static_cast<DecoderEvents*>(events);
    return kOk;
}
int DecGetOutputFormat(void* self, void* md) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    if (!d->codec) return kFail;
    AMediaFormat* fmt = AMediaCodec_getOutputFormat(d->codec);
    if (!fmt) return kFail;
    FillMediaData(MD(md), fmt, true);
    return kOk;
#else
    return kFail;
#endif
}
int DecStart(void* self) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    if (!d->codec) return kFail;
    return AMediaCodec_start(d->codec) == AMEDIA_OK ? kOk : kFail;
#else
    return kFail;
#endif
}
int DecStop(void* self) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    if (d->codec) AMediaCodec_stop(d->codec);
    std::lock_guard<std::mutex> lk(d->m);
    d->inputs.clear();
    d->outputs.clear();
    d->held.clear();
#endif
    return kOk;
}
int DecFlush(void* self) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    if (d->codec) AMediaCodec_flush(d->codec);
    {
        std::lock_guard<std::mutex> lk(d->m);
        d->inputs.clear();
        d->outputs.clear();
        d->held.clear();
    }
    // In async mode the codec must be restarted after a flush.
    if (d->codec) AMediaCodec_start(d->codec);
#endif
    return kOk;
}
int DecReset(void* self) { return DecFlush(self); }
int DecAcquireInputBufferIndex(void* self, int* index, int* remaining) {
    Decoder* d = Dec(self);
    std::lock_guard<std::mutex> lk(d->m);
    if (d->inputs.empty()) {
        *index = -1;
        *remaining = 0;
        return kFail;
    }
    *index = d->inputs.front();
    d->inputs.pop_front();
    *remaining = int(d->inputs.size());
    return kOk;
}
int DecGetInputBuffer(void* self, int index, void* buffer) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    size_t cap = 0;
    if (index < 0) return kFail;
    uint8_t* p = d->codec ? AMediaCodec_getInputBuffer(d->codec, size_t(index), &cap) : nullptr;
    if (!p) return kFail;
    SetBuffer(buffer, p, cap, 0, 0);
    return kOk;
#else
    return kFail;
#endif
}
int DecSendInputBufferForDecode(void* self, int index, int offset, int size, int64_t time_us, uint32_t flags) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    if (!d->codec || index < 0) return kFail;
    return AMediaCodec_queueInputBuffer(d->codec, size_t(index), off_t(offset), size_t(size), uint64_t(time_us),
                                        flags) == AMEDIA_OK
               ? kOk
               : kFail;
#else
    return kFail;
#endif
}
int DecAcquireOutputBufferIndex(void* self, int* index, int64_t* time_us, uint32_t* flags, int* size) {
    Decoder* d = Dec(self);
    std::lock_guard<std::mutex> lk(d->m);
    const auto now = std::chrono::steady_clock::now();
    if (now - d->last_report > std::chrono::seconds(3)) {
        d->last_report = now;
        LOGI(kTag, "decoder type %d: %llu decoded, %llu taken by the game, %llu released, %zu waiting, %zu held, "
             "%llu empty polls, last pts %.2f s", d->type, (unsigned long long)d->produced,
             (unsigned long long)d->acquired, (unsigned long long)d->released, d->outputs.size(), d->held.size(),
             (unsigned long long)d->polls_empty, d->last_pts / 1e6);
    }
    if (d->outputs.empty()) {
        d->polls_empty++;
        *index = -1;
        return kFail;
    }
    OutputItem o = d->outputs.front();
    d->outputs.pop_front();
    d->held[o.index] = o;
    d->acquired++;
    d->last_pts = o.time_us;
    *index = o.index;
    *time_us = o.time_us;
    *flags = o.flags;
    *size = o.size;
    return kOk;
}
int DecGetOutputBuffer(void* self, int index, void* buffer) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    size_t cap = 0;
    if (index < 0) return kFail;
    uint8_t* p = d->codec ? AMediaCodec_getOutputBuffer(d->codec, size_t(index), &cap) : nullptr;
    if (!p) return kFail;
    OutputItem o{};
    {
        std::lock_guard<std::mutex> lk(d->m);
        auto it = d->held.find(index);
        if (it != d->held.end()) o = it->second;
    }
    // Video into memory the game supplied (output format 3: RGBA, e.g. a texture's storage): convert the
    // decoded NV12 picture (BT.709, limited range) into it. Otherwise hand out the decoder's own buffer.
    BufferState& gb = Buf(buffer);
    if (d->type == 1 && gb.from_game && gb.data && d->width > 0 && d->height > 0) {
        int w, h, stride, slice, cl, ct;
        {
            std::lock_guard<std::mutex> lk(d->m);
            w = d->width;
            h = d->height;
            stride = d->stride;
            slice = d->slice_height;
            cl = d->crop_left;
            ct = d->crop_top;
        }
        const size_t pitch = size_t(w) * 4;
        const size_t need_src = size_t(o.offset) + size_t(stride) * size_t(slice) * 3 / 2;
        if (size_t(cap) < need_src) {
            if (!d->logged_convert) {
                d->logged_convert = true;
                LOGW(kTag, "decoded frame smaller than expected (%zu < %zu bytes)", size_t(cap), need_src);
            }
            SetBuffer(buffer, p, cap, o.offset, o.size);
            return kOk;
        }
        // Convert NV12 (BT.709, limited range) to RGBA: straight into the game's memory when it is big enough,
        // else into a picture that is written into the texture owning that memory (the game passes its video
        // texture's storage with a token capacity, as the hardware decoder writes it directly). That write
        // runs on the render thread, between command lists: the backend reads (uploads) that memory while it
        // executes them, and a half-written frame shows up as blocks of two different frames.
        const bool direct = gb.capacity >= pitch * size_t(h);
        // Frame buffers are reused once the backend has uploaded (released) them.
        static std::mutex pool_m;
        static std::vector<std::shared_ptr<std::vector<uint8_t>>> pool;
        std::shared_ptr<std::vector<uint8_t>> frame;
        if (!direct) {
            std::lock_guard<std::mutex> lk(pool_m);
            for (auto& f : pool)
                if (f.use_count() == 1 && f->size() == pitch * size_t(h)) {
                    frame = f;
                    break;
                }
            if (!frame && pool.size() < 8) {
                frame = std::make_shared<std::vector<uint8_t>>(pitch * size_t(h));
                pool.push_back(frame);
            }
            if (!frame) frame = std::make_shared<std::vector<uint8_t>>(pitch * size_t(h));
        }
        uint8_t* dst = direct ? gb.data : frame->data();
        const auto t0 = std::chrono::steady_clock::now();
        ConvertNv12(p + o.offset, stride, slice, cl, ct, w, h, dst, pitch);
        RecordConvertTime(std::chrono::steady_clock::now() - t0);
        // Into a texture: the picture goes to the backend, which uploads it at the texture's next use (writing
        // the texture's memory instead raced with the backend reading it: blocks of two frames).
        bool ok = direct;
        if (!direct)
            ok = dsr::hle::nvn::SubmitTextureFrame(uint64_t(reinterpret_cast<uintptr_t>(gb.data)), frame, w, h);
        if (!d->logged_convert) {
            d->logged_convert = true;
            LOGI(kTag, "video frame %dx%d (stride %d, slice height %d) -> game memory %p (capacity %zu): %s", w, h,
                 stride, slice, static_cast<void*>(gb.data), gb.capacity,
                 direct ? "converted in place" : ok ? "handed to that texture" : "NO RGBA TEXTURE THERE");
        }
        if (ok) {
            if (direct) {
                gb.offset = 0;
                gb.size = int(pitch * size_t(h));
            }
            return kOk;
        }
    }
    SetBuffer(buffer, p, cap, o.offset, o.size);
    return kOk;
#else
    return kFail;
#endif
}
int DecGetOutputBufferSize(void* self, int* a, int* b) {
    *a = *b = 0;
    return kFail;
}
int DecReleaseOutputBufferIndex(void* self, int index) {
#if defined(__ANDROID__)
    Decoder* d = Dec(self);
    {
        std::lock_guard<std::mutex> lk(d->m);
        d->held.erase(index);
        d->released++;
    }
    if (d->codec && index >= 0) AMediaCodec_releaseOutputBuffer(d->codec, size_t(index), false);
#endif
    return kOk;
}
int DecGetLastError(void*) { return kOk; }
int DecGetOutputBuffer3(void* self, int index, void* buffer, void* md) {
    int r = DecGetOutputBuffer(self, index, buffer);
    if (r == kOk && md) DecGetOutputFormat(self, md);
    return r;
}
int DecGetOutputBufferFormat(void* self, int, void* md) { return DecGetOutputFormat(self, md); }

// ---- vtables ---------------------------------------------------------------------------------------
template <size_t N>
const void* BuildVtable(void* const (&fns)[N]) {
    auto* vt = new uint64_t[N + 2]();
    for (size_t i = 0; i < N; i++) vt[i + 2] = reinterpret_cast<uint64_t>(dsr::loader::MakeDynamicThunk(fns[i], 0));
    return vt + 2;
}
#define F(f) reinterpret_cast<void*>(&f)
const void* ExtractorVptr() {
    static void* const fns[] = {F(ExDtor), F(ExDtorDelete), F(ExSetDataSourcePath), F(ExSetDataSourceMem),
                                F(ExGetTrackCount), F(ExGetTrackConfiguration), F(ExSelectTrack),
                                F(ExUnselectTrack), F(ExSeekTo), F(ExGetTrackIndexForAvailableData), F(ExRead),
                                F(ExAdvance), F(ExGetSampleTime), F(ExGetSampleMetaData), F(ExGetBufferedRange),
                                F(ExSetDataSourceReader)};
    static const void* vptr = BuildVtable(fns);
    return vptr;
}
const void* DecoderVptr() {
    static void* const fns[] = {F(DecDtor), F(DecDtorDelete), F(DecConfigure), F(DecRegisterClientEvents),
                                F(DecGetOutputFormat), F(DecStart), F(DecStop), F(DecReset), F(DecFlush),
                                F(DecAcquireInputBufferIndex), F(DecGetInputBuffer), F(DecSendInputBufferForDecode),
                                F(DecAcquireOutputBufferIndex), F(DecGetOutputBuffer), F(DecGetOutputBufferSize),
                                F(DecReleaseOutputBufferIndex), F(DecGetLastError), F(DecGetOutputBuffer3),
                                F(DecGetOutputBufferFormat)};
    static const void* vptr = BuildVtable(fns);
    return vptr;
}
#undef F
}  // namespace

// ===== imports ===========================================================================================
HLE_FUNC(_ZN5movie12SetAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_, void, movie_SetAllocator,
         (const void*, const void*, const void*, void*)) {}
HLE_FUNC(_ZN5movie11SetCoreMaskEm, void, movie_SetCoreMask, (uint64_t)) {}

HLE_FUNC(_ZN5movie9ExtractorC1ENS_13ContainerTypeE, void, movie_ExtractorCtor, (void* self, int container)) {
    Trace("Extractor::Extractor");
    *static_cast<const void**>(self) = ExtractorVptr();
    ImplOf<Extractor>(self) = new Extractor;
}
HLE_FUNC(_ZN5movie7DecoderC1ENS_11DecoderTypeENS_19DecoderOutputFormatENS_11DecoderModeE, void, movie_DecoderCtor,
         (void* self, int type, int format, int mode)) {
    LOGI(kTag, "Decoder::Decoder(type %d, output format %d, mode %d)", type, format, mode);
    *static_cast<const void**>(self) = DecoderVptr();
    auto* d = new Decoder;
    d->type = type;
    d->output_format = format;
    d->mode = mode;
    ImplOf<Decoder>(self) = d;
}
HLE_FUNC(_ZN5movie7DecoderD1Ev, void, movie_DecoderDtor, (void* self)) { DecDtor(self); }
HLE_FUNC(_ZN5movie7Decoder9ConfigureEPNS_9MediaDataE, int, movie_DecoderConfigure, (void* s, void* m)) {
    return DecConfigure(s, m);
}
HLE_FUNC(_ZN5movie7Decoder20RegisterClientEventsEPNS_13DecoderEventsE, int, movie_DecoderRegister, (void* s, void* e)) {
    return DecRegisterClientEvents(s, e);
}
HLE_FUNC(_ZN5movie7Decoder5StartEv, int, movie_DecoderStart, (void* s)) { return DecStart(s); }
HLE_FUNC(_ZN5movie7Decoder4StopEv, int, movie_DecoderStop, (void* s)) { return DecStop(s); }
HLE_FUNC(_ZN5movie7Decoder23AcquireInputBufferIndexEPiS1_, int, movie_DecoderAcqIn, (void* s, int* a, int* b)) {
    return DecAcquireInputBufferIndex(s, a, b);
}
HLE_FUNC(_ZN5movie7Decoder14GetInputBufferEiPNS_6BufferE, int, movie_DecoderGetIn, (void* s, int i, void* b)) {
    return DecGetInputBuffer(s, i, b);
}
HLE_FUNC(_ZN5movie7Decoder24SendInputBufferForDecodeEiiilj, int, movie_DecoderSend,
         (void* s, int i, int o, int n, int64_t t, uint32_t f)) {
    return DecSendInputBufferForDecode(s, i, o, n, t, f);
}
HLE_FUNC(_ZN5movie7Decoder24AcquireOutputBufferIndexEPiPlPjS1_, int, movie_DecoderAcqOut,
         (void* s, int* i, int64_t* t, uint32_t* f, int* c)) {
    return DecAcquireOutputBufferIndex(s, i, t, f, c);
}
HLE_FUNC(_ZN5movie7Decoder15GetOutputBufferEiPNS_6BufferE, int, movie_DecoderGetOut, (void* s, int i, void* b)) {
    return DecGetOutputBuffer(s, i, b);
}
HLE_FUNC(_ZN5movie7Decoder24ReleaseOutputBufferIndexEi, int, movie_DecoderRelease, (void* s, int i)) {
    return DecReleaseOutputBufferIndex(s, i);
}

// MediaData
HLE_FUNC(_ZN5movie9MediaDataC1Ev, void, movie_MediaDataCtor, (void* self)) {
    std::lock_guard<std::mutex> lk(g_md_mutex);
    g_media_data[self] = MediaData{};
}
HLE_FUNC(_ZN5movie9MediaDataC1ERKS0_, void, movie_MediaDataCopy, (void* self, const void* other)) {
    std::lock_guard<std::mutex> lk(g_md_mutex);
    MediaData copy = g_media_data[other];
    g_media_data[self] = copy;
}
HLE_FUNC(_ZN5movie9MediaDataD1Ev, void, movie_MediaDataDtor, (void* self)) {
    std::lock_guard<std::mutex> lk(g_md_mutex);
    g_media_data.erase(self);
}
HLE_FUNC(_ZNK5movie9MediaData10FindStringEPKcPS2_, bool, movie_MediaDataFindString,
         (const void* self, const char* key, const char** out)) {
    std::lock_guard<std::mutex> lk(g_md_mutex);
    auto& md = g_media_data[self];
    auto it = md.strings.find(key ? key : "");
    if (it == md.strings.end()) return false;
    *out = it->second.c_str();
    return true;
}
HLE_FUNC(_ZNK5movie9MediaData9FindInt32EPKcPi, bool, movie_MediaDataFindInt32,
         (const void* self, const char* key, int* out)) {
    std::lock_guard<std::mutex> lk(g_md_mutex);
    auto& md = g_media_data[self];
    auto it = md.ints.find(key ? key : "");
    if (it == md.ints.end()) return false;
    *out = int(it->second);
    return true;
}

// Buffer
HLE_FUNC(_ZN5movie6BufferC1Ev, void, movie_BufferCtor, (void* self)) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    g_buffers[self] = BufferState{};
}
HLE_FUNC(_ZN5movie6BufferC1Em, void, movie_BufferCtorSize, (void* self, size_t n)) {
    BufferState s;
    s.data = static_cast<uint8_t*>(calloc(1, n ? n : 1));
    s.capacity = n;
    s.owned = true;
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    g_buffers[self] = s;
}
HLE_FUNC(_ZN5movie6BufferD1Ev, void, movie_BufferDtor, (void* self)) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    auto it = g_buffers.find(self);
    if (it == g_buffers.end()) return;
    if (it->second.owned) free(it->second.data);
    g_buffers.erase(it);
}
HLE_FUNC(_ZN5movie6Buffer18SetDataAndCapacityEPvm, void, movie_BufferSetData, (void* self, void* data, size_t cap)) {
    SetBuffer(self, static_cast<uint8_t*>(data), cap, 0, 0);
    Buf(self).from_game = data != nullptr;
}
HLE_FUNC(_ZN5movie6Buffer8SetRangeEii, void, movie_BufferSetRange, (void* self, int offset, int size)) {
    BufferState& s = Buf(self);
    s.offset = offset;
    s.size = size;
}
HLE_FUNC(_ZNK5movie6Buffer4BaseEv, void*, movie_BufferBase, (const void* self)) { return Buf(self).data; }
HLE_FUNC(_ZNK5movie6Buffer4SizeEv, size_t, movie_BufferSize, (const void* self)) { return size_t(Buf(self).size); }
HLE_FUNC(_ZNK5movie6Buffer6OffsetEv, size_t, movie_BufferOffset, (const void* self)) {
    return size_t(Buf(self).offset);
}
