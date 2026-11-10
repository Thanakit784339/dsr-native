// nn::audio AudioOut (FMOD's output path). Buffers appended by the game are played in order by a host
// thread; finished buffers move to a "released" list and the game's SystemEvent is signaled.
//
// Backends: AAudio on Android (blocking writes), or a real-time "null" sink elsewhere.
#include <algorithm>
#include <cstdlib>
#include <condition_variable>
#include <vector>
#include <cstring>
#include <deque>
#include <mutex>
#include <chrono>
#include <string>
#include <thread>

#include "common/log.h"
#include "common/platform.h"
#include "hle/os/os.h"
#include "hle/registry.h"
#include "runtime/runtime.h"

#if defined(__ANDROID__)
#include <aaudio/AAudio.h>
#endif

namespace {
constexpr const char* kTag = "audio";
constexpr uint32_t kOk = 0;
constexpr uint32_t kAudioInUnavailable = 153u | (1u << 9);
constexpr int kSampleRate = 48000;
constexpr int kChannels = 2;
constexpr int32_t kBufferFrames = 3072;  // host stream buffer: at least 64 ms at 48 kHz

// Guest AudioOutBuffer: our layout within the SDK's struct.
struct GuestBuffer {
    void* data;
    size_t buffer_size;
    size_t data_size;
    void* reserved;
};

// nn::audio::AudioOutParameter
struct AudioOutParameter {
    int32_t sample_rate;
    uint16_t channel_count;
    uint16_t reserved;
    uint32_t magic;
};

class AudioOut {
public:
    AudioOut(void* system_event, int channels) : event_(system_event), channels_(channels > 0 ? channels : 2) {}
    ~AudioOut() { Stop(); }

    bool Open() {
#if defined(__ANDROID__)
        AAudioStreamBuilder* b = nullptr;
        if (AAudio_createStreamBuilder(&b) != AAUDIO_OK) return false;
        AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_I16);
        AAudioStreamBuilder_setChannelCount(b, kChannels);
        AAudioStreamBuilder_setSampleRate(b, kSampleRate);
        AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
        aaudio_result_t r = AAudioStreamBuilder_openStream(b, &stream_);
        AAudioStreamBuilder_delete(b);
        if (r != AAUDIO_OK) {
            LOGE(kTag, "AAudio openStream failed: %s", AAudio_convertResultToText(r));
            stream_ = nullptr;
        } else {
            // A low-latency stream defaults to a buffer of a burst or two (a few ms): the game's buffer is
            // released only once its samples fit in there, so any delay of the game's audio thread (busy
            // CPU, e.g. while a video plays) runs the device dry and the sound stutters. Keep ~64 ms queued.
            // Never below two bursts (a normal-mode stream has bursts of tens of ms: one is not enough).
            const int32_t cap = AAudioStream_getBufferCapacityInFrames(stream_);
            const int32_t burst = AAudioStream_getFramesPerBurst(stream_);
            const int32_t want = std::max({kBufferFrames, 2 * burst, AAudioStream_getBufferSizeInFrames(stream_)});
            AAudioStream_setBufferSizeInFrames(stream_, std::min(want, cap));
            LOGI(kTag, "AAudio stream: %d Hz, burst %d frames, buffer %d of %d frames, %s", AAudioStream_getSampleRate(stream_),
                 AAudioStream_getFramesPerBurst(stream_), AAudioStream_getBufferSizeInFrames(stream_), cap,
                 AAudioStream_getPerformanceMode(stream_) == AAUDIO_PERFORMANCE_MODE_LOW_LATENCY ? "low latency"
                                                                                                   : "normal");
        }
#endif
        return true;
    }

    void Start() {
        if (running_) return;
        running_ = true;
#if defined(__ANDROID__)
        if (stream_) AAudioStream_requestStart(stream_);
#endif
        worker_ = std::thread([this] { Run(); });
#if defined(__ANDROID__)
        if (Smooth()) pacer_ = std::thread([this] { Pace(); });
#endif
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_) return;
            running_ = false;
        }
        cv_.notify_all();
        pace_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (pacer_.joinable()) pacer_.join();
#if defined(__ANDROID__)
        if (stream_) {
            AAudioStream_requestStop(stream_);
            AAudioStream_close(stream_);
            stream_ = nullptr;
        }
#endif
    }

    void Append(GuestBuffer* b) {
        {
            std::lock_guard<std::mutex> lk(m_);
            queued_.push_back(b);
        }
        cv_.notify_all();
    }

    GuestBuffer* PopReleased() {
        std::lock_guard<std::mutex> lk(m_);
        if (released_.empty()) return nullptr;
        GuestBuffer* b = released_.front();
        released_.pop_front();
        return b;
    }

private:
    void Run() {
        while (true) {
            GuestBuffer* b;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return !running_ || !queued_.empty(); });
                if (!running_) return;
                b = queued_.front();
            }
            if (dsr::platform::IsPaused()) {
                // In the background: the stream is paused (silence, no audio focus held) and nothing more is
                // played or released, so the game's audio thread waits too.
#if defined(__ANDROID__)
                if (stream_) AAudioStream_requestPause(stream_);
#endif
                dsr::platform::WaitWhilePaused();
#if defined(__ANDROID__)
                if (stream_) AAudioStream_requestStart(stream_);
#endif
            }
            const size_t frames = b->data_size / (sizeof(int16_t) * size_t(channels_));
            const int16_t* pcm = ToStereo(static_cast<const int16_t*>(b->data), frames);
            Report(pcm, frames);
            Play(pcm, frames);
            written_ += frames;
            bool now_released = false;
            {
                std::lock_guard<std::mutex> lk(m_);
                queued_.pop_front();
                if (Smooth()) {
                    accepted_.push_back({b, written_});
                } else {
                    released_.push_back(b);
                    now_released = true;
                }
            }
            if (now_released) Released(1);
            pace_cv_.notify_all();
        }
    }

    // Buffer release timing. Android takes audio in bursts (tens of ms of samples at once on a normal-mode
    // stream), and the game was told a buffer was done the moment its samples were accepted: release times came
    // in clumps. Games that time video by audio progress (the opening movie) then showed frames in bursts and
    // held them. Accepted buffers are now released when a smoothly interpolated read position (the device's
    // read counter, advanced at the sample rate between its steps, capped at one burst ahead) says the device
    // has room for them: the same moment on average, without the clumps.
    struct Accepted {
        GuestBuffer* b;
        uint64_t end;  // frames written up to and including this buffer
    };
    bool Smooth() const {
#if defined(__ANDROID__)
        return stream_ && smooth_;
#else
        return false;
#endif
    }
    void Released(int n) {
        TraceReleases(n);
        if (event_) dsr::hle::os::SignalSystemEvent(event_);
    }
    void Pace() {
#if defined(__ANDROID__)
        int64_t last_read = -1;
        std::chrono::steady_clock::time_point last_change{};
        const int32_t burst = std::max<int32_t>(1, AAudioStream_getFramesPerBurst(stream_));
        const int32_t buffer = std::max<int32_t>(burst, AAudioStream_getBufferSizeInFrames(stream_));
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                pace_cv_.wait_for(lk, std::chrono::milliseconds(2));
                if (!running_) {
                    for (const Accepted& a : accepted_) released_.push_back(a.b);
                    accepted_.clear();
                    return;
                }
                if (accepted_.empty()) continue;
            }
            const auto now = std::chrono::steady_clock::now();
            const int64_t read = AAudioStream_getFramesRead(stream_);
            if (read != last_read) {
                last_read = read;
                last_change = now;
            }
            const double since = std::chrono::duration<double>(now - last_change).count();
            const double pos = double(read) + std::min(double(burst), since * kSampleRate);
            int n = 0;
            {
                std::lock_guard<std::mutex> lk(m_);
                while (!accepted_.empty() && double(accepted_.front().end) <= pos + buffer) {
                    released_.push_back(accepted_.front().b);
                    accepted_.pop_front();
                    n++;
                }
            }
            if (n) Released(n);
        }
#endif
    }
    // Diagnostics (movie stutter): intervals between the first releases of each stream, logged once.
    std::chrono::steady_clock::time_point trace_last_{};
    std::string trace_;
    int traced_ = 0;
    void TraceReleases(int n) {
        if (traced_ >= 120) return;
        const auto now = std::chrono::steady_clock::now();
        if (traced_ > 0) {
            char b[24];
            snprintf(b, sizeof(b), " %.0f%s", std::chrono::duration<double, std::milli>(now - trace_last_).count(),
                     n > 1 ? ("x" + std::to_string(n)).c_str() : "");
            trace_ += b;
        }
        trace_last_ = now;
        if (++traced_ == 120)
            LOGI(kTag, "audio out (%d ch) buffer releases, ms apart (xN: N at once):%s", channels_, trace_.c_str());
    }

    // The game may open the output with 1, 2 or 6 (5.1: FL FR FC LFE BL BR) channels; the host stream is
    // always stereo.
    const int16_t* ToStereo(const int16_t* in, size_t frames) {
        if (channels_ == 2) return in;
        stereo_.resize(frames * 2);
        for (size_t i = 0; i < frames; i++) {
            const int16_t* f = in + i * size_t(channels_);
            int l, r;
            if (channels_ == 1) {
                l = r = f[0];
            } else if (channels_ >= 6) {
                const float c = 0.7071f * f[2], lfe = 0.5f * f[3];
                l = int(f[0] + c + 0.7071f * f[4] + lfe);
                r = int(f[1] + c + 0.7071f * f[5] + lfe);
            } else {
                l = f[0];
                r = f[1];
            }
            stereo_[2 * i] = int16_t(std::clamp(l, -32768, 32767));
            stereo_[2 * i + 1] = int16_t(std::clamp(r, -32768, 32767));
        }
        return stereo_.data();
    }

    // Every ~10 s of audio, log how many buffers were played and their peak level, so "no sound" can
    // be told apart from "the game produces only silence".
    void Report(const int16_t* pcm, size_t frames) {
        for (size_t i = 0; i < frames * 2; i++) peak_ = std::max(peak_, std::abs(int(pcm[i])));
        frames_played_ += frames;
        buffers_played_++;
        if (frames_played_ >= next_report_) {
            int32_t xruns = 0;
#if defined(__ANDROID__)
            if (stream_) xruns = AAudioStream_getXRunCount(stream_);
#endif
            LOGI(kTag, "audio out (%d ch): %llu buffers, %.1f s played, peak level %d/32767 (%s), %d underruns",
                 channels_, (unsigned long long)buffers_played_, double(frames_played_) / kSampleRate, peak_,
                 peak_ < 64 ? "silence" : "sound", xruns);
            next_report_ = frames_played_ + 10 * kSampleRate;
            peak_ = 0;
        }
    }

    void Play(const int16_t* pcm, size_t frames) {
#if defined(__ANDROID__)
        if (stream_) {
            while (frames) {
                aaudio_result_t n = AAudioStream_write(stream_, pcm, int32_t(frames), 100'000'000);
                if (n < 0) {
                    LOGW(kTag, "AAudio write failed: %s", AAudio_convertResultToText(n));
                    break;
                }
                pcm += size_t(n) * kChannels;
                frames -= size_t(n);
            }
            return;
        }
#endif
        (void)pcm;
        // Null sink: consume in real time.
        std::this_thread::sleep_for(std::chrono::microseconds(frames * 1'000'000 / kSampleRate));
    }

    void* event_;
    int channels_;
    std::vector<int16_t> stereo_;
    uint64_t frames_played_ = 0, buffers_played_ = 0, next_report_ = 0;
    int peak_ = 0;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<GuestBuffer*> queued_, released_;
    std::deque<Accepted> accepted_;  // written to the device, not yet released (smoothing)
    std::condition_variable pace_cv_;
    std::thread pacer_;
    uint64_t written_ = 0;
    bool smooth_ = dsr::GetRuntime().config.audio_smooth;
    std::thread worker_;
    bool running_ = false;
#if defined(__ANDROID__)
    AAudioStream* stream_ = nullptr;
#endif
};

AudioOut* Out(void* guest) { return *static_cast<AudioOut**>(guest); }
}  // namespace

HLE_FUNC(_ZN2nn5audio27InitializeAudioOutParameterEPNS0_17AudioOutParameterE, void, audio_InitializeAudioOutParameter,
         (AudioOutParameter * p)) {
    p->sample_rate = kSampleRate;
    p->channel_count = kChannels;
    p->reserved = 0;
    p->magic = 0x1234abcd;
}
// AudioOut* is the game's object; its first 8 bytes hold our host pointer. SystemEvent* (nn::os::SystemEvent
// class) begins with a SystemEventType.
HLE_FUNC(_ZN2nn5audio19OpenDefaultAudioOutEPNS0_8AudioOutEPNS_2os11SystemEventERKNS0_17AudioOutParameterE, uint32_t,
         audio_OpenDefaultAudioOut, (void* out, void* system_event, const AudioOutParameter* p)) {
    LOGI(kTag, "OpenDefaultAudioOut(rate=%d ch=%u)", p->sample_rate, p->channel_count);
    if (system_event) dsr::hle::os::InitSystemEvent(system_event, true, false);
    auto* ao = new AudioOut(system_event, p->channel_count);
    ao->Open();
    *static_cast<AudioOut**>(out) = ao;
    return kOk;
}
HLE_FUNC(_ZN2nn5audio13StartAudioOutEPNS0_8AudioOutE, uint32_t, audio_StartAudioOut, (void* out)) {
    Out(out)->Start();
    return kOk;
}
HLE_FUNC(_ZN2nn5audio12StopAudioOutEPNS0_8AudioOutE, void, audio_StopAudioOut, (void* out)) { Out(out)->Stop(); }
HLE_FUNC(_ZN2nn5audio13CloseAudioOutEPNS0_8AudioOutE, void, audio_CloseAudioOut, (void* out)) {
    delete Out(out);
    *static_cast<AudioOut**>(out) = nullptr;
}
HLE_FUNC(_ZN2nn5audio21SetAudioOutBufferInfoEPNS0_14AudioOutBufferEPvmm, void, audio_SetAudioOutBufferInfo,
         (GuestBuffer * b, void* data, size_t size, size_t data_size)) {
    b->data = data;
    b->buffer_size = size;
    b->data_size = data_size;
    b->reserved = nullptr;
}
HLE_FUNC(_ZN2nn5audio28GetAudioOutBufferDataPointerEPKNS0_14AudioOutBufferE, void*, audio_GetAudioOutBufferDataPointer,
         (const GuestBuffer* b)) {
    return b->data;
}
HLE_FUNC(_ZN2nn5audio20AppendAudioOutBufferEPNS0_8AudioOutEPNS0_14AudioOutBufferE, bool, audio_AppendAudioOutBuffer,
         (void* out, GuestBuffer* b)) {
    Out(out)->Append(b);
    return true;
}
HLE_FUNC(_ZN2nn5audio25GetReleasedAudioOutBufferEPNS0_8AudioOutE, GuestBuffer*, audio_GetReleasedAudioOutBuffer,
         (void* out)) {
    return Out(out)->PopReleased();
}

// ---- AudioIn (microphone): report no devices ----------------------------------------------------------
HLE_FUNC(_ZN2nn5audio12ListAudioInsEPNS0_11AudioInInfoEi, int, audio_ListAudioIns, (void*, int)) { return 0; }
HLE_FUNC(_ZN2nn5audio26InitializeAudioInParameterEPNS0_16AudioInParameterE, void, audio_InitializeAudioInParameter,
         (AudioOutParameter * p)) {
    p->sample_rate = kSampleRate;
    p->channel_count = 1;
}
HLE_FUNC(_ZN2nn5audio11OpenAudioInEPNS0_7AudioInEPKcRKNS0_16AudioInParameterE, uint32_t, audio_OpenAudioIn,
         (void*, const char*, const void*)) {
    return kAudioInUnavailable;
}
HLE_FUNC(_ZN2nn5audio12StartAudioInEPNS0_7AudioInE, uint32_t, audio_StartAudioIn, (void*)) { return kAudioInUnavailable; }
HLE_FUNC(_ZN2nn5audio11StopAudioInEPNS0_7AudioInE, void, audio_StopAudioIn, (void*)) {}
HLE_FUNC(_ZN2nn5audio12CloseAudioInEPNS0_7AudioInE, void, audio_CloseAudioIn, (void*)) {}
HLE_FUNC(_ZN2nn5audio24GetReleasedAudioInBufferEPNS0_7AudioInE, void*, audio_GetReleasedAudioInBuffer, (void*)) {
    return nullptr;
}
HLE_FUNC(_ZN2nn5audio19AppendAudioInBufferEPNS0_7AudioInEPNS0_13AudioInBufferE, bool, audio_AppendAudioInBuffer,
         (void*, void*)) {
    return false;
}
HLE_ALIAS(_ZN2nn5audio20SetAudioInBufferInfoEPNS0_13AudioInBufferEPvmm, audio_SetAudioOutBufferInfo);
HLE_ALIAS(_ZN2nn5audio27GetAudioInBufferDataPointerEPKNS0_13AudioInBufferE, audio_GetAudioOutBufferDataPointer);
HLE_FUNC(_ZN2nn5audio20GetAudioInSampleRateEPKNS0_7AudioInE, int, audio_GetAudioInSampleRate, (const void*)) {
    return kSampleRate;
}
HLE_FUNC(_ZN2nn5audio22GetAudioInChannelCountEPKNS0_7AudioInE, int, audio_GetAudioInChannelCount, (const void*)) {
    return 1;
}
HLE_FUNC(_ZN2nn5audio22GetAudioInSampleFormatEPKNS0_7AudioInE, int, audio_GetAudioInSampleFormat, (const void*)) {
    return 2;  // SampleFormat_PcmInt16
}
