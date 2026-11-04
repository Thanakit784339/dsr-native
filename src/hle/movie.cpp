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

#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <string>
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
        // else into a scratch picture that is written into the texture owning that memory (the game passes
        // its video texture's storage with a token capacity, as the hardware decoder writes it directly).
        const bool direct = gb.capacity >= pitch * size_t(h);
        static thread_local std::vector<uint8_t> scratch;
        uint8_t* dst = direct ? gb.data : (scratch.resize(pitch * size_t(h)), scratch.data());
        const uint8_t* y_plane = p + o.offset;
        const uint8_t* uv_plane = y_plane + size_t(stride) * size_t(slice);
        for (int yy = 0; yy < h; yy++) {
            const uint8_t* yr = y_plane + size_t(yy + ct) * size_t(stride) + cl;
            const uint8_t* uvr = uv_plane + size_t((yy + ct) / 2) * size_t(stride) + (cl & ~1);
            uint8_t* out = dst + size_t(yy) * pitch;
            for (int xx = 0; xx < w; xx++) {
                const int c = 298 * (int(yr[xx]) - 16);
                const int u = int(uvr[xx & ~1]) - 128, v = int(uvr[(xx & ~1) + 1]) - 128;
                int r = (c + 459 * v + 128) >> 8;              // 1.793
                int g = (c - 55 * u - 136 * v + 128) >> 8;     // 0.213, 0.533
                int b = (c + 541 * u + 128) >> 8;              // 2.112
                out[xx * 4 + 0] = uint8_t(r < 0 ? 0 : r > 255 ? 255 : r);
                out[xx * 4 + 1] = uint8_t(g < 0 ? 0 : g > 255 ? 255 : g);
                out[xx * 4 + 2] = uint8_t(b < 0 ? 0 : b > 255 ? 255 : b);
                out[xx * 4 + 3] = 255;
            }
        }
        bool ok = direct;
        if (!direct)
            ok = dsr::hle::nvn::WriteTextureAtAddress(uint64_t(reinterpret_cast<uintptr_t>(gb.data)), dst, w, h, pitch);
        if (!d->logged_convert) {
            d->logged_convert = true;
            LOGI(kTag, "video frame %dx%d (stride %d, slice height %d) -> game memory %p (capacity %zu): %s", w, h,
                 stride, slice, static_cast<void*>(gb.data), gb.capacity,
                 direct ? "converted in place" : ok ? "written into the texture at that address" : "NO TEXTURE THERE");
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
