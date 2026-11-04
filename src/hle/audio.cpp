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
#include <thread>

#include "common/log.h"
#include "hle/os/os.h"
#include "hle/registry.h"

#if defined(__ANDROID__)
#include <aaudio/AAudio.h>
#endif

namespace {
constexpr const char* kTag = "audio";
constexpr uint32_t kOk = 0;
constexpr uint32_t kAudioInUnavailable = 153u | (1u << 9);
constexpr int kSampleRate = 48000;
constexpr int kChannels = 2;

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
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!running_) return;
            running_ = false;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
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
            const size_t frames = b->data_size / (sizeof(int16_t) * size_t(channels_));
            const int16_t* pcm = ToStereo(static_cast<const int16_t*>(b->data), frames);
            Report(pcm, frames);
            Play(pcm, frames);
            {
                std::lock_guard<std::mutex> lk(m_);
                queued_.pop_front();
                released_.push_back(b);
            }
            if (event_) dsr::hle::os::SignalSystemEvent(event_);
        }
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
            LOGI(kTag, "audio out: %llu buffers, %.1f s played, peak level %d/32767 (%s)",
                 (unsigned long long)buffers_played_, double(frames_played_) / kSampleRate, peak_,
                 peak_ < 64 ? "silence" : "sound");
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
