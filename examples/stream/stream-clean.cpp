// Real-time speech recognition with clean text output and signal-based completion
// Supports context-based corrections and immediate termination on signal

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

// Global flag for signal-based termination
std::atomic<bool> g_force_transcribe(false);
std::atomic<bool> g_is_transcribing(false);

// Signal handler for SIGUSR1 - triggers immediate transcription
void handle_sigusr1(int sig) {
    if (sig == SIGUSR1) {
        g_force_transcribe = true;
    }
}

// Check for stdin commands (non-blocking)
bool check_stdin_command() {
    // Make stdin non-blocking
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    
    char buffer[256];
    ssize_t bytes_read = read(STDIN_FILENO, buffer, sizeof(buffer) - 1);
    
    // Restore blocking mode
    fcntl(STDIN_FILENO, F_SETFL, flags);
    
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        std::string cmd(buffer);
        
        // Remove newline
        if (!cmd.empty() && cmd.back() == '\n') {
            cmd.pop_back();
        }
        
        // Check for 'f' command
        if (cmd == "f" || cmd == "F") {
            return true;
        }
    }
    
    return false;
}

struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 500;     // Reduced for better responsiveness
    int32_t length_ms  = 5000;    // Shorter chunks for lower latency
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    
    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;
    
    bool translate     = false;
    bool no_fallback   = false;
    bool use_gpu       = true;
    bool flash_attn    = false;
    
    std::string language  = "auto";
    std::string model     = "models/ggml-base.en.bin";
};

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            fprintf(stderr, "\n");
            fprintf(stderr, "usage: %s [options]\n", argv[0]);
            fprintf(stderr, "\n");
            fprintf(stderr, "options:\n");
            fprintf(stderr, "  -h,       --help          show this help message and exit\n");
            fprintf(stderr, "  -t N,     --threads N     number of threads to use during computation\n");
            fprintf(stderr, "            --step N        audio step size in milliseconds\n");
            fprintf(stderr, "            --length N      audio length in milliseconds\n");
            fprintf(stderr, "            --keep N        audio to keep from previous step in ms\n");
            fprintf(stderr, "  -c ID,    --capture ID    capture device ID\n");
            fprintf(stderr, "  -vth N,   --vad-thold N   voice activity detection threshold\n");
            fprintf(stderr, "  -fth N,   --freq-thold N  high-pass frequency cutoff\n");
            fprintf(stderr, "  -tr,      --translate     translate from source language to english\n");
            fprintf(stderr, "  -nf,      --no-fallback   do not use temperature fallback while decoding\n");
            fprintf(stderr, "  -l LANG,  --language LANG spoken language\n");
            fprintf(stderr, "  -m FNAME, --model FNAME   model path\n");
            fprintf(stderr, "  -ng,      --no-gpu        disable GPU inference\n");
            fprintf(stderr, "  -fa,      --flash-attn    flash attention during inference\n");
            fprintf(stderr, "\n");
            fprintf(stderr, "Signal control:\n");
            fprintf(stderr, "  - Send SIGUSR1 to trigger immediate transcription\n");
            fprintf(stderr, "  - Type 'finish' + Enter to trigger immediate transcription\n");
            fprintf(stderr, "\n");
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }
    
    return true;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    
    // Install signal handler
    signal(SIGUSR1, handle_sigusr1);
    
    whisper_params params;
    
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }
    
    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);
    
    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;
    
    // Always use sliding window mode for continuous operation
    const bool use_vad = false;
    
    // init audio
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    
    audio.resume();
    
    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
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
    
    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
    
    std::vector<whisper_token> prompt_tokens;
    
    // Output initialization
    fprintf(stderr, "\n[Ready] Type 'f' for full transcription. Signal SIGUSR1 for Electron integration.\n\n");
    fflush(stderr);
    
    int n_iter = 0;
    bool is_running = true;
    
    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;
    
    // Keep track of accumulated transcription
    std::string accumulated_text;
    std::string full_session_text;  // All text from this session
    bool has_speech = false;
    int silence_duration_ms = 0;
    int no_speech_iterations = 0;
    
    // main audio loop
    while (is_running) {
        // Check for termination signals
        is_running = sdl_poll_events();
        
        if (!is_running) {
            break;
        }
        
        // Check for stdin command
        bool force_transcribe_stdin = check_stdin_command();
        
        // Check if we should force transcription
        bool should_force = g_force_transcribe.exchange(false) || force_transcribe_stdin;
        
        // Collect audio samples
        while (true) {
            audio.get(params.step_ms, pcmf32_new);
            
            if ((int) pcmf32_new.size() > 2*n_samples_step) {
                fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                audio.clear();
                continue;
            }
            
            if ((int) pcmf32_new.size() >= n_samples_step) {
                audio.clear();
                break;
            }
            
            // Check signals while waiting
            if (g_force_transcribe || check_stdin_command()) {
                should_force = true;
                g_force_transcribe = false;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        const int n_samples_new = pcmf32_new.size();
        
        // Take up to params.length_ms audio from previous iteration
        const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));
        
        pcmf32.resize(n_samples_new + n_samples_take);
        
        for (int i = 0; i < n_samples_take; i++) {
            pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
        }
        
        memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));
        
        pcmf32_old = pcmf32;
        
        // Simple VAD - check if we have speech
        float energy = 0.0f;
        for (const auto& sample : pcmf32_new) {
            energy += sample * sample;
        }
        energy = sqrtf(energy / pcmf32_new.size());
        
        bool speech_detected = energy > params.vad_thold * 0.01f; // Adjust threshold
        
        if (speech_detected) {
            has_speech = true;
            silence_duration_ms = 0;
        } else {
            silence_duration_ms += params.step_ms;
        }
        
        // Decide whether to transcribe
        bool should_transcribe = false;
        bool is_final_transcription = false;
        
        if (should_force) {
            // 'f' command - get everything immediately
            should_transcribe = true;
            is_final_transcription = true;
            fprintf(stderr, "\n");
        } else if (has_speech && silence_duration_ms > 500) {
            // Natural pause detected after speech - show incremental update
            should_transcribe = true;
        } else if (has_speech && n_iter > 0 && (n_iter % 4) == 0) {
            // Regular interval transcription while speaking
            should_transcribe = true;
        }
        
        if (should_transcribe) {
            g_is_transcribing = true;
            
            // Run the inference
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            
            wparams.print_progress   = false;
            wparams.print_special    = false;
            wparams.print_realtime   = false;
            wparams.print_timestamps = false;
            wparams.translate        = params.translate;
            wparams.single_segment   = true;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            
            wparams.audio_ctx        = params.audio_ctx;
            
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
            
            // Use context for better accuracy
            wparams.prompt_tokens    = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = prompt_tokens.size();
            
            // Temporarily suppress stderr to hide language detection output
            int stderr_fd = dup(STDERR_FILENO);
            FILE* null_file = fopen("/dev/null", "w");
            dup2(fileno(null_file), STDERR_FILENO);
            
            int result = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());
            
            // Restore stderr
            dup2(stderr_fd, STDERR_FILENO);
            close(stderr_fd);
            fclose(null_file);
            
            if (result != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                g_is_transcribing = false;
                return 6;
            }
            
            // Get the transcribed text
            const int n_segments = whisper_full_n_segments(ctx);
            std::string current_text;
            
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);
                // Skip blank audio segments
                if (text && strcmp(text, "[BLANK_AUDIO]") != 0) {
                    current_text += text;
                }
            }
            
            // Output the transcribed text
            if (!current_text.empty()) {
                // Trim whitespace
                current_text.erase(0, current_text.find_first_not_of(" \n\r\t"));
                current_text.erase(current_text.find_last_not_of(" \n\r\t") + 1);
                
                if (!current_text.empty()) {
                    if (is_final_transcription) {
                        // Final transcription - output everything clean
                        printf("\n[FINAL TRANSCRIPTION]\n");
                        printf("%s%s\n", full_session_text.c_str(), current_text.c_str());
                        printf("[END TRANSCRIPTION]\n");
                        fflush(stdout);
                        
                        // Reset for next session
                        full_session_text.clear();
                        accumulated_text.clear();
                        has_speech = false;
                        prompt_tokens.clear();
                        pcmf32_old.clear();
                    } else {
                        // Real-time update - show just the new part
                        printf("\r");  // Carriage return to overwrite current line
                        
                        // Show accumulated text for this segment
                        std::string display_text = accumulated_text + current_text;
                        printf("%s", display_text.c_str());
                        
                        // Add padding to clear any previous longer text
                        printf("%*s", std::max(0, 80 - (int)display_text.length()), "");
                        
                        fflush(stdout);
                        
                        accumulated_text = display_text;
                        
                        // If we detect end of sentence, commit it to full session
                        char last_char = current_text.back();
                        if ((last_char == '.' || last_char == '!' || last_char == '?') && silence_duration_ms > 300) {
                            full_session_text += accumulated_text + " ";
                            accumulated_text.clear();
                            printf("\n");  // New line for next sentence
                            fflush(stdout);
                        }
                    }
                }
            }
            
            if (!is_final_transcription) {
                // Update prompt tokens for context
                prompt_tokens.clear();
                
                for (int i = 0; i < n_segments; ++i) {
                    const int token_count = whisper_full_n_tokens(ctx, i);
                    for (int j = 0; j < token_count; ++j) {
                        prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                    }
                }
                
                // Keep part of the audio for next iteration
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
                
                // Reset speech detection after long silence
                if (silence_duration_ms > 2000) {
                    has_speech = false;
                    accumulated_text.clear();
                }
            }
            
            ++n_iter;
            g_is_transcribing = false;
        }
    }
    
    audio.pause();
    
    whisper_free(ctx);
    
    return 0;
}