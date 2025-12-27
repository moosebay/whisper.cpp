// Stream audio transcription from stdin
// Reads raw 16-bit mono PCM at 16kHz from stdin and transcribes in real-time
//
// Usage: sysaudiod | ./whisper-stream-stdin -m model.bin

#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool use_gpu       = true;

    std::string language = "en";
    std::string model    = "models/ggml-base.en.bin";
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                 arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                 arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                 arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-nf"  || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"  || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"   || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-ng"  || arg == "--no-gpu")        { params.use_gpu       = false; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(1);
        }
    }
    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "Reads raw 16-bit signed little-endian mono PCM at 16kHz from stdin.\n");
    fprintf(stderr, "Example: ./sysaudiod | %s -m models/ggml-base.en.bin\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads\n", params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n", params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n", params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n", params.keep_ms);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum tokens per audio chunk\n", params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 = all)\n", params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] VAD threshold\n", params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n", params.freq_thold);
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] disable temperature fallback\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n", params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between chunks\n", params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "\n");
}

// Audio buffer that reads from stdin in a background thread
class stdin_audio {
public:
    stdin_audio(int sample_rate) : m_sample_rate(sample_rate), m_running(false) {}

    bool start() {
#if defined(_WIN32)
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        m_running = true;
        m_thread = std::thread(&stdin_audio::read_loop, this);
        return true;
    }

    void stop() {
        m_running = false;
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    // Get up to `ms` milliseconds of audio, returns number of samples
    size_t get(int ms, std::vector<float> & result) {
        std::lock_guard<std::mutex> lock(m_mutex);

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_buffer.size()) {
            n_samples = m_buffer.size();
        }

        result.resize(n_samples);
        if (n_samples > 0) {
            std::copy(m_buffer.begin(), m_buffer.begin() + n_samples, result.begin());
            m_buffer.erase(m_buffer.begin(), m_buffer.begin() + n_samples);
        }

        return n_samples;
    }

    size_t available() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.size();
    }

    bool is_running() const { return m_running; }

private:
    void read_loop() {
        std::vector<int16_t> buf(1024);

        while (m_running) {
            size_t n = fread(buf.data(), sizeof(int16_t), buf.size(), stdin);
            if (n == 0) {
                if (feof(stdin)) {
                    fprintf(stderr, "\n[stdin closed]\n");
                    m_running = false;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Convert int16 to float32 and add to buffer
            std::lock_guard<std::mutex> lock(m_mutex);
            for (size_t i = 0; i < n; i++) {
                m_buffer.push_back(static_cast<float>(buf[i]) / 32768.0f);
            }
        }
    }

    int m_sample_rate;
    std::atomic<bool> m_running;
    std::thread m_thread;
    mutable std::mutex m_mutex;
    std::vector<float> m_buffer;
};

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (!whisper_params_parse(argc, argv, params)) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms, params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;

    const int n_new_line = std::max(1, params.length_ms / params.step_ms - 1);

    params.no_timestamps = true;
    params.max_tokens    = 0;

    // Initialize whisper
    fprintf(stderr, "whisper: loading model '%s'\n", params.model.c_str());

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 1;
    }

    // Start reading from stdin
    stdin_audio audio(WHISPER_SAMPLE_RATE);
    if (!audio.start()) {
        fprintf(stderr, "error: failed to start audio reader\n");
        whisper_free(ctx);
        return 1;
    }

    fprintf(stderr, "whisper: model loaded, waiting for audio from stdin...\n");
    fprintf(stderr, "whisper: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec)\n",
            n_samples_step,
            float(n_samples_step) / WHISPER_SAMPLE_RATE,
            float(n_samples_len) / WHISPER_SAMPLE_RATE,
            float(n_samples_keep) / WHISPER_SAMPLE_RATE);
    fprintf(stderr, "\n");

    std::vector<float> pcmf32(n_samples_len, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new;
    std::vector<whisper_token> prompt_tokens;

    int n_iter = 0;

    while (audio.is_running()) {
        // Wait until we have enough samples
        while (audio.available() < (size_t)n_samples_step && audio.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (!audio.is_running()) break;

        // Get new audio samples
        audio.get(params.step_ms, pcmf32_new);

        if ((int)pcmf32_new.size() < n_samples_step) {
            continue;
        }

        const int n_samples_new = pcmf32_new.size();

        // Take up to params.length_ms audio from previous iteration
        const int n_samples_take = std::min((int)pcmf32_old.size(),
                                            std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        pcmf32.resize(n_samples_new + n_samples_take);

        for (int i = 0; i < n_samples_take; i++) {
            pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
        }

        memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new * sizeof(float));

        pcmf32_old = pcmf32;

        // Run inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = false;
            wparams.single_segment   = true;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.audio_ctx        = params.audio_ctx;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0 : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "whisper: failed to process audio\n");
                break;
            }

            // Print result
            {
                printf("\33[2K\r");

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);
                    printf("%s", text);
                    fflush(stdout);
                }
            }

            ++n_iter;

            if ((n_iter % n_new_line) == 0) {
                printf("\n");

                // Keep part of the audio for next iteration
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Update prompt tokens for context
                if (!params.no_context) {
                    prompt_tokens.clear();
                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
        }
    }

    printf("\n");
    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
