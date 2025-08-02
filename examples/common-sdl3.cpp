#include "common-sdl.h"

#include <cstdio>

// SDL3 compatibility definitions
#undef SDL_TRUE
#undef SDL_FALSE  
#undef AUDIO_F32
#define SDL_TRUE true
#define SDL_FALSE false
#define AUDIO_F32 SDL_AUDIO_F32

audio_async::audio_async(int len_ms) {
    m_len_ms = len_ms;
    m_running = false;
}

audio_async::~audio_async() {
    if (m_dev_id_in) {
        SDL_CloseAudioDevice(m_dev_id_in);
    }
}

bool audio_async::init(int capture_id, int sample_rate) {
    SDL_SetLogPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if (!SDL_Init(SDL_INIT_AUDIO)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
        return false;
    }

    // SDL3 doesn't have AUDIO_RESAMPLING_MODE hint anymore

    // List audio devices
    {
        SDL_AudioDeviceID *devices = SDL_GetAudioRecordingDevices(nullptr);
        if (devices) {
            int nDevices = 0;
            for (int i = 0; devices[i] != 0; i++) {
                nDevices++;
            }
            fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
            
            for (int i = 0; devices[i] != 0; i++) {
                const char *name = SDL_GetAudioDeviceName(devices[i]);
                fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, name ? name : "Unknown");
            }
            SDL_free(devices);
        }
    }

    SDL_AudioSpec capture_spec;
    SDL_zero(capture_spec);
    
    capture_spec.freq     = sample_rate;
    capture_spec.format   = SDL_AUDIO_F32;
    capture_spec.channels = 1;

    // SDL3 uses SDL_OpenAudioDevice differently
    SDL_AudioDeviceID device_to_open = 0;
    const char *device_name = nullptr;
    
    if (capture_id >= 0) {
        SDL_AudioDeviceID *devices = SDL_GetAudioRecordingDevices(nullptr);
        if (devices && capture_id < 100) { // sanity check
            int idx = 0;
            for (int i = 0; devices[i] != 0; i++) {
                if (idx == capture_id) {
                    device_to_open = devices[i];
                    device_name = SDL_GetAudioDeviceName(devices[i]);
                    fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__, capture_id, device_name ? device_name : "Unknown");
                    break;
                }
                idx++;
            }
            SDL_free(devices);
        }
    }
    
    // Open the audio device
    if (device_to_open != 0) {
        m_dev_id_in = SDL_OpenAudioDevice(device_to_open, &capture_spec);
    } else {
        fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
        m_dev_id_in = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_RECORDING, &capture_spec);
    }

    if (!m_dev_id_in) {
        fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());
        return false;
    }

    // Create audio stream for format conversion if needed
    m_stream = SDL_CreateAudioStream(&capture_spec, &capture_spec);
    if (!m_stream) {
        fprintf(stderr, "%s: couldn't create audio stream: %s!\n", __func__, SDL_GetError());
        return false;
    }

    // Bind the stream to the opened device
    if (!SDL_BindAudioStream(m_dev_id_in, m_stream)) {
        fprintf(stderr, "%s: couldn't bind audio stream: %s!\n", __func__, SDL_GetError());
        SDL_DestroyAudioStream(m_stream);
        m_stream = nullptr;
        return false;
    }

    fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, m_dev_id_in);
    fprintf(stderr, "%s:     - sample rate:       %d\n", __func__, sample_rate);
    fprintf(stderr, "%s:     - format:            SDL_AUDIO_F32\n", __func__);
    fprintf(stderr, "%s:     - channels:          1\n", __func__);

    m_sample_rate = sample_rate;
    m_audio.resize((m_sample_rate * m_len_ms) / 1000);

    return true;
}

bool audio_async::resume() {
    if (!m_dev_id_in || !m_stream) {
        fprintf(stderr, "%s: no audio device to resume!\n", __func__);
        return false;
    }

    if (m_running) {
        fprintf(stderr, "%s: already running!\n", __func__);
        return false;
    }

    if (!SDL_ResumeAudioDevice(m_dev_id_in)) {
        fprintf(stderr, "%s: couldn't resume audio device: %s!\n", __func__, SDL_GetError());
        return false;
    }

    m_running = true;

    return true;
}

bool audio_async::pause() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to pause!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: already paused!\n", __func__);
        return false;
    }

    if (!SDL_PauseAudioDevice(m_dev_id_in)) {
        fprintf(stderr, "%s: couldn't pause audio device: %s!\n", __func__, SDL_GetError());
        return false;
    }

    m_running = false;

    return true;
}

bool audio_async::clear() {
    if (!m_dev_id_in || !m_stream) {
        fprintf(stderr, "%s: no audio device to clear!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_audio_pos = 0;
        m_audio_len = 0;
        
        // Clear the stream
        SDL_ClearAudioStream(m_stream);
    }

    return true;
}

// This is no longer a callback, we need to poll the stream
void audio_async::callback(uint8_t * stream, int len) {
    if (!m_running) {
        return;
    }

    const size_t n_samples = len / sizeof(float);

    m_audio_new.resize(n_samples);
    memcpy(m_audio_new.data(), stream, n_samples * sizeof(float));

    //fprintf(stderr, "%s: %zu samples, pos %zu, len %zu\n", __func__, n_samples, m_audio_pos, m_audio_len);

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;

            memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
            memcpy(&m_audio[0], &stream[n0 * sizeof(float)], (n_samples - n0) * sizeof(float));

            m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
            m_audio_len = m_audio.size();
        } else {
            memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));

            m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
            m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
        }
    }
}

void audio_async::get(int ms, std::vector<float> & result) {
    if (!m_dev_id_in || !m_stream) {
        fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
        return;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }

    // First, read any available audio from the stream
    {
        // Read up to 1 second of audio at a time
        const int max_samples = m_sample_rate;
        std::vector<float> buffer(max_samples);
        
        int read_samples = SDL_GetAudioStreamData(m_stream, buffer.data(), max_samples * sizeof(float));
        if (read_samples > 0) {
            callback((uint8_t*)buffer.data(), read_samples);
        }
    }

    // Now get the requested amount of audio
    result.clear();

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);

        int s0 = m_audio_pos - n_samples;
        if (s0 < 0) {
            s0 += m_audio.size();
        }

        if (s0 + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - s0;

            memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
        } else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
        }
    }
}

bool sdl_poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
                return false;
                break;
            default:
                break;
        }
    }

    return true;
}