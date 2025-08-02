# whisper.cpp/examples/stream

This is a naive example of performing real-time inference on audio from your microphone.
The `whisper-stream` tool samples the audio every half a second and runs the transcription continously.
More info is available in [issue #10](https://github.com/ggerganov/whisper.cpp/issues/10).

```bash
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a80f-28ba83be7d09.mp4

## Sliding window mode with VAD

Setting the `--step` argument to `0` enables the sliding window mode:

```bash
 ./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 6 --step 0 --length 30000 -vth 0.6
```

In this mode, the tool will transcribe only after some speech activity is detected. A very
basic VAD detector is used, but in theory a more sophisticated approach can be added. The
`-vth` argument determines the VAD threshold - higher values will make it detect silence more often.
It's best to tune it to the specific use case, but a value around `0.6` should be OK in general.
When silence is detected, it will transcribe the last `--length` milliseconds of audio and output
a transcription block that is suitable for parsing.

## Building

The `whisper-stream` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2
# On Debian based linux distributions:
sudo apt-get install libsdl2-dev

# On Fedora Linux:
sudo dnf install SDL2 SDL2-devel

# Install SDL2 on Mac OS
brew install sdl2

cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release

./build/bin/whisper-stream
```

## Web version

This tool can also run in the browser: [examples/stream.wasm](/examples/stream.wasm)

---

# whisper-stream-clean: Real-time Stream with Static SDL3

This is an enhanced version that provides continuous real-time transcription with context-based corrections and creates a self-contained binary without SDL runtime dependencies.

## Features

- Real-time transcription that appears as you speak
- Context-based corrections using Whisper's capabilities
- Immediate transcription on demand (press 'f' or send SIGUSR1)
- Self-contained binary - no SDL3 runtime required
- Ultra-low latency (< 400ms)
- Clean output suitable for integration with Electron/Node.js

## Building with Static SDL3

### Prerequisites

1. Clone whisper.cpp repository
2. Have CMake and a C++ compiler installed

### Build SDL3 as Static Library

```bash
# From whisper.cpp root directory
mkdir -p deps && cd deps

# Clone SDL3
git clone https://github.com/libsdl-org/SDL.git SDL2
cd SDL2

# Configure SDL3 for static build
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DSDL_SHARED=OFF \
  -DSDL_STATIC=ON \
  -DSDL_TEST=OFF \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"

# Build SDL3
make -j8
cd ../../..
```

### Build whisper-stream-clean

```bash
# From whisper.cpp root directory
mkdir build && cd build

# Configure with SDL2 enabled (which uses our SDL3)
cmake .. -DWHISPER_SDL2=ON

# Build the stream-clean binary
make whisper-stream-clean -j8

# The binary will be at: build/bin/whisper-stream-clean
```

## Usage

```bash
./whisper-stream-clean \
  -m "path/to/model.bin" \
  -c 0 \                    # Microphone device ID
  -l auto \                 # Auto-detect language
  --step 300 \              # Process every 300ms
  --length 3000 \           # 3 second chunks
  --keep 200 \              # Keep 200ms context
  -vth 0.5                  # Voice activity threshold
```

### Controls

- **Real-time mode**: Transcription appears automatically as you speak
- **Press 'f'**: Get immediate full transcription with all corrections
- **SIGUSR1**: Send signal for programmatic control (for Electron integration)

### Example Output

```
[Ready] Type 'f' for full transcription. Signal SIGUSR1 for Electron integration.

Hello, this is a test of real-time transcription.
It updates as you speak and corrects based on context.

[FINAL TRANSCRIPTION]
Hello, this is a test of real-time transcription. It updates as you speak and corrects based on context.
[END TRANSCRIPTION]
```

## Integration with Electron

The binary accepts SIGUSR1 signals for programmatic control:

```javascript
// In your Node.js/Electron app
const { spawn } = require('child_process');

const whisper = spawn('./whisper-stream-clean', [
  '-m', modelPath,
  '-c', '0',
  // ... other args
]);

// Trigger immediate transcription
process.kill(whisper.pid, 'SIGUSR1');

// Parse the output
whisper.stdout.on('data', (data) => {
  const output = data.toString();
  if (output.includes('[FINAL TRANSCRIPTION]')) {
    // Handle final transcription
  }
});
```

## Important Notes

1. **Self-contained**: Once built, the binary includes SDL3 statically linked. No runtime dependencies needed.
2. **Cross-platform**: The static build approach works on macOS. For other platforms, adjust the SDL3 build flags.
3. **Model**: You need a Whisper model file (e.g., ggml-small-q8_0.bin)
4. **Performance**: Achieves < 400ms latency on modern hardware

## Troubleshooting

- **No audio devices found**: Check microphone permissions
- **Model loading failed**: Verify model path and format
- **High latency**: Try reducing --length parameter or using a smaller model
- **Build errors**: Ensure SDL3 is built as static library with matching architecture