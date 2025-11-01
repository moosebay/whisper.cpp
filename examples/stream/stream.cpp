// Real-time speech recognition of input from a microphone
//
// Variant: RECORD → BUFFER → SINGLE FINAL TRANSCRIPTION
// -----------------------------------------------------
// - While recording (SIGUSR2): we ONLY collect audio into one big buffer.
//   We do NOT append text from partial / sliding windows, so no duplicates.
// - When recording stops (SIGUSR1): we run ONE whisper_full(...) over the
//   ENTIRE buffered audio and print once.
//
// This gives the best accuracy (like transcribing a WAV) and avoids the
// duplicated segments that come from overlapping windows.
//

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <signal.h>

// Global flags for signal-based control
std::atomic<bool> g_is_recording(false);
std::atomic<bool> g_force_transcribe(false);

// Signal handler for SIGUSR1 - stops recording and triggers final transcription
void handle_sigusr1(int sig) {
    if (sig == SIGUSR1 && g_is_recording) {
        g_force_transcribe = true;
        g_is_recording = false;
    }
}

// Signal handler for SIGUSR2 - starts recording
void handle_sigusr2(int sig) {
    if (sig == SIGUSR2 && !g_is_recording) {
        g_is_recording = true;
        fprintf(stderr, "[Recording started]\n");
    }
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;   // kept for CLI compatibility, not used in final pass
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 0;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = true;   // we do a final pass, can be changed with CLI
    bool tinydiarize   = false;
    bool save_audio    = false;
    bool use_gpu       = true;
    bool flash_attn    = false;

    std::string language  = "auto";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds (not used in final mode)\n", params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks (used only if you re-enable streaming)\n", params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    // Install signal handlers
    signal(SIGUSR1, handle_sigusr1);
    signal(SIGUSR2, handle_sigusr2);

    whisper_params params;
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // init audio
    audio_async * audio = nullptr;

    // init whisper
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    // big buffer that will hold ALL audio for the current recording session
    std::vector<float> recorded_audio;

    std::ofstream fout;
    if (!params.fname_out.empty()) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "error: failed to open output file '%s'\n", params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    if (params.save_audio) {
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";
        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }

    printf("[Start speaking]\n");
    fflush(stdout);
    fprintf(stderr, "[Model loaded] Send SIGUSR2 to start recording, SIGUSR1 to stop and transcribe.\n");

    bool is_running     = true;
    bool was_recording  = false;

    while (is_running) {
        // handle SDL / Ctrl+C
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        // just started recording
        if (g_is_recording && !was_recording) {
            if (audio == nullptr) {
                audio = new audio_async(params.length_ms);
                if (!audio->init(params.capture_id, WHISPER_SAMPLE_RATE)) {
                    fprintf(stderr, "Failed to initialize audio device!\n");
                    delete audio;
                    audio = nullptr;
                    g_is_recording = false;
                    continue;
                }
                audio->resume();
            }
            recorded_audio.clear();
            was_recording = true;
        }

        // just stopped recording
        if (!g_is_recording && was_recording) {
            fprintf(stderr, "[Recording stopped - transcribing...]\n");

            // final transcription over the FULL buffer
            if (!recorded_audio.empty()) {
                whisper_full_params wparams =
                    whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                                                                     : WHISPER_SAMPLING_GREEDY);

                wparams.print_progress   = false;
                wparams.print_special    = params.print_special;
                wparams.print_realtime   = false;
                wparams.print_timestamps = !params.no_timestamps;
                wparams.translate        = params.translate;
                wparams.single_segment   = false;
                wparams.max_tokens       = params.max_tokens;
                wparams.language         = params.language.c_str();
                wparams.n_threads        = params.n_threads;
                wparams.beam_search.beam_size = params.beam_size;
                wparams.audio_ctx        = params.audio_ctx;
                wparams.tdrz_enable      = params.tinydiarize;
                wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

                auto t0 = std::chrono::high_resolution_clock::now();
                if (whisper_full(ctx, wparams, recorded_audio.data(), recorded_audio.size()) != 0) {
                    fprintf(stderr, "error: final transcription failed\n");
                } else {
                    std::string final_text;
                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const char * text = whisper_full_get_segment_text(ctx, i);
                        if (params.no_timestamps) {
                            final_text += text;
                        } else {
                            const int64_t t0_ms = whisper_full_get_segment_t0(ctx, i);
                            const int64_t t1_ms = whisper_full_get_segment_t1(ctx, i);
                            final_text += "[" + to_timestamp(t0_ms, false) + " --> " + to_timestamp(t1_ms, false) + "] ";
                            final_text += text;
                            final_text += "\n";
                        }
                    }

                    // print once
                    printf("%s\n", final_text.c_str());
                    fflush(stdout);

                    if (fout.is_open()) {
                        fout << final_text << std::endl;
                    }
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                fprintf(stderr, "[Final transcription took %ld ms]\n", (long)dur);
            }

            // teardown audio
            if (audio != nullptr) {
                audio->pause();
                delete audio;
                audio = nullptr;
            }

            was_recording     = false;
            g_force_transcribe = false;
            continue;
        }

        // if not recording, rest
        if (!g_is_recording || audio == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // we ARE recording: keep reading small chunks and append to big buffer
        std::vector<float> chunk;
        audio->get(200, chunk); // 200 ms
        if (!chunk.empty()) {
            recorded_audio.insert(recorded_audio.end(), chunk.begin(), chunk.end());
            if (params.save_audio) {
                wavWriter.write(chunk.data(), chunk.size());
            }
        }

        // small sleep to avoid busy loop
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (audio != nullptr) {
        audio->pause();
        delete audio;
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
