# Jughead — Always-on voice assistant for Linux

Jughead is a hands-free, wake-word-activated voice assistant that listens continuously and executes commands based on spoken intents. It uses WebRTC VAD for efficient voice detection and `faster-whisper` for local speech-to-text.

## Features

| Category | Commands |
|---|---|
| **Apps** | Open/switch to Chrome, Zed, Ghostty, Nautilus, Spotify |
| **Web search** | Google, YouTube, ChatGPT |
| **Folders** | Projects, Downloads, Documents |
| **Volume** | Set %, raise/lower, mute/unmute/toggle |
| **Time/date** | Current time, date, or both |
| **Docker** | Status, start/stop Docker, docker compose up |
| **System** | Lock screen, suspend, system status report (CPU/memory/disk/temp/battery) |
| **Macros** | "start my day", "work mode", "chill mode" |
| **Other** | GitHub profile/repos, WhatsApp Web, LeetCode |
| **Assistant** | Go to sleep / wake up |

## Requirements

- **Linux** with PulseAudio (for volume) and ALSA (for TTS playback)
- **Piper TTS** binary and a voice model (e.g. `en_US-lessac-medium.onnx`)
- Optional: `hyprctl` (Hyprland), `swaymsg` (Sway), or `wmctrl` (X11) for window focus

### Python dependencies

```
faster-whisper sounddevice webrtcvad-wheels rapidfuzz psutil piper-tts
```

## Setup

```bash
git clone <repo> jughead
cd jughead
python3 -m venv env
source env/bin/activate
pip install faster-whisper sounddevice webrtcvad-wheels rapidfuzz psutil piper-tts
```

Install a Piper voice model:
```bash
mkdir -p ~/.local/share/piper
# Download en_US-lessac-medium.onnx into ~/.local/share/piper/
```

## Usage

```bash
./env/bin/python main.py
```

Say **"Friday"** to wake the assistant, then speak your command.

## Configuration

Edit `main.py:22-37` to change the wake word, audio device, model size, terminal/browser preference, and GitHub username.

## License

MIT
