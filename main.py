#!/usr/bin/env python3
"""Always-on voice assistant: VAD-gated faster-whisper command runner."""

import collections
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import psutil
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import fuzz
from rapidfuzz import process as fuzzproc

# ---------- Config ----------
WAKE_WORD = "friday"
MODEL_SIZE = "small.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_LEVEL = 2
SILENCE_END = 0.8
MAX_UTTERANCE = 10.0
TERMINAL = "ghostty"
BROWSER = "google-chrome-stable"
PIPER_MODEL = os.path.expanduser("~/.local/share/piper/en_US-lessac-medium.onnx")
PIPER_BIN = os.path.expanduser("~/Projects/jughead/env/bin/piper")
FUZZY_THRESHOLD = 80  # 0-100, lower = more forgiving
GITHUB_USERNAME = "vansh16-code"  # <- change this


FOLDERS = {
    "projects": "~/Projects",
    "downloads": "~/Downloads",
    "documents": "~/Documents",
}

SHORTCUTS = {
    "lock screen": "loginctl lock-session",
    "suspend system": "systemctl suspend",
}

# Docker commands with voice feedback and terminal display
DOCKER_COMMANDS = {
    "docker status": {
        "cmd": "docker ps",
        "speak": "Checking Docker status",
        "terminal": True,
    },
    "start docker": {
        "cmd": "systemctl start docker",
        "speak": "Starting Docker",
        "terminal": True,
    },
    "stop docker": {
        "cmd": "systemctl stop docker",
        "speak": "Stopping Docker",
        "terminal": True,
    },
    "docker compose up": {
        "cmd": "docker compose up -d",
        "speak": "Starting Docker Compose",
        "terminal": True,
    },
}

# Multi-step macros: spoken trigger -> list of sub-commands dispatched in order
# Each string is handled exactly like a spoken command (after the wake word)
MACROS = {
    "start my day": [
        "open dsa",  # leetcode
        "open editor",  # zed editor
        "open terminal",  # ghostty
    ],
    "work mode": [
        "open editor",
        "open terminal",
        "open chrome",
    ],
    "chill mode": [
        "open spotify",
        "open chrome",
    ],
}

# spoken name -> (window class to focus, command to launch if not running)
APPS = {
    "chrome": ("google-chrome", BROWSER),
    "editor": ("zeditor", "zeditor"),
    "terminal": ("com.mitchellh.ghostty", TERMINAL),
    "files": ("org.gnome.Nautilus", "nautilus"),
    "spotify": ("spotify", "spotify"),
}

# words whisper commonly mishears, corrected before matching
VOCAB = [
    "open",
    "search",
    "focus",
    "terminal",
    "zed",
    "chrome",
    "youtube",
    "files",
    "downloads",
    "documents",
    "projects",
    "docker",
    "status",
    "report",
    "system",
    "lock",
    "screen",
    "suspend",
    "chatgpt",
    "gpt",
    "sleep",
    "wake",
    "spotify",
    "compose",
    "start",
    "stop",
    "github",
    "profile",
    "repositories",
    "whatsapp",
    "leetcode",
    # volume
    "volume",
    "mute",
    "unmute",
    "louder",
    "quieter",
    # time & date
    "time",
    "date",
    "today",
    # macros
    "macro",
    "mode",
    "chill",
    "work",
]

# ---------- State ----------
asleep = False
suppress_until = 0.0  # ignore mic while TTS is playing


# ---------- Helpers ----------
def run(cmd):
    subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def notify(msg):
    run(["notify-send", "-a", "Voice Assistant", "Voice Assistant", msg])


def expand(path_or_alias):
    return os.path.expanduser(FOLDERS.get(path_or_alias.strip(), path_or_alias.strip()))


def beep(freq=880, dur=0.12):
    t = np.linspace(0, dur, int(44100 * dur), False)
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = np.linspace(1, 0, len(tone)).astype(np.float32)
    sd.play(tone * fade, 44100)


def speak(text):
    """TTS via piper, suppresses mic while playing."""
    global suppress_until

    def _tts():
        global suppress_until
        try:
            p = subprocess.run(
                [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", "/tmp/va_tts.wav"],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            if p.returncode != 0:
                notify(text)
                return
            dur = os.path.getsize("/tmp/va_tts.wav") / (22050 * 2)
            suppress_until = time.time() + dur + 0.5
            subprocess.run(["aplay", "-q", "/tmp/va_tts.wav"])
        except FileNotFoundError:
            notify(text)  # piper or aplay missing, fall back to notification

    threading.Thread(target=_tts, daemon=True).start()


def run_docker_command(name: str):
    """Run docker command in visible terminal with voice feedback."""
    if name not in DOCKER_COMMANDS:
        return False

    config = DOCKER_COMMANDS[name]
    speak(config["speak"])

    if config.get("terminal", False):
        # Run in a new terminal window so user can see output
        run(
            [
                TERMINAL,
                "-e",
                "bash",
                "-c",
                f"{config['cmd']}; echo; read -p 'Press Enter to close...'",
            ]
        )
    else:
        run(config["cmd"])

    notify(f"Ran: {name}")
    return True


def system_report():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    bat = psutil.sensors_battery()
    temps = psutil.sensors_temperatures()
    temp = next(iter(temps.values()))[0].current if temps else None
    spoken = (
        f"CPU at {cpu:.0f} percent. Memory at {mem.percent:.0f} percent. "
        f"Disk at {disk.percent:.0f} percent."
    )
    if temp:
        spoken += f" Temperature {temp:.0f} degrees."
    if bat:
        spoken += f" Battery at {bat.percent:.0f} percent."
    notify(spoken)
    speak(spoken)


def web_search(engine, query):
    q = query.strip().replace(" ", "+")
    urls = {
        "chrome": f"https://www.google.com/search?q={q}",
        "youtube": f"https://www.youtube.com/results?search_query={q}",
        "chatgpt": f"https://chatgpt.com/?q={q}",
    }
    run([BROWSER, urls[engine]])


# ---------- Window management ----------
def focus_or_launch(name):
    name = name.strip()
    match = fuzzproc.extractOne(name, APPS.keys(), scorer=fuzz.ratio)
    if not match or match[1] < FUZZY_THRESHOLD:
        notify(f"Unknown app: {name}")
        return
    wclass, launch_cmd = APPS[match[0]]

    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        r = subprocess.run(
            ["hyprctl", "dispatch", "focuswindow", f"class:(?i){wclass}"],
            capture_output=True,
            text=True,
        )
        if "no window" not in r.stdout.lower() and r.returncode == 0:
            return
    elif os.environ.get("SWAYSOCK"):
        r = subprocess.run(
            ["swaymsg", f'[app_id="{wclass}"] focus'], capture_output=True
        )
        if r.returncode == 0:
            return
    else:  # X11
        r = subprocess.run(["wmctrl", "-x", "-a", wclass], capture_output=True)
        if r.returncode == 0:
            return
    run(launch_cmd)  # not running -> launch it


# ---------- Fuzzy correction ----------
def fuzzy_correct(text):
    """Correct each misheard word against known vocabulary."""
    out = []
    for word in text.split():
        if len(word) < 3 or word in VOCAB:
            out.append(word)
            continue
        match = fuzzproc.extractOne(word, VOCAB, scorer=fuzz.ratio)
        out.append(match[0] if match and match[1] >= FUZZY_THRESHOLD else word)
    return " ".join(out)


def has_wake_word(text):
    return any(fuzz.ratio(w, WAKE_WORD) >= 75 for w in text.split()[:4])


def strip_wake_word(text):
    words = text.split()
    for i, w in enumerate(words[:4]):
        if fuzz.ratio(w, WAKE_WORD) >= 75:
            return " ".join(words[i + 1 :])
    return text


# ---------- Volume ----------
def set_volume(level: int):
    level = max(0, min(100, level))
    run(f"pactl set-sink-volume @DEFAULT_SINK@ {level}%")
    speak(f"Volume set to {level} percent.")


def change_volume(delta: int):
    sign = "+" if delta >= 0 else "-"
    run(f"pactl set-sink-volume @DEFAULT_SINK@ {sign}{abs(delta)}%")
    direction = "up" if delta >= 0 else "down"
    speak(f"Volume {direction} by {abs(delta)} percent.")


def mute():
    run("pactl set-sink-mute @DEFAULT_SINK@ 1")
    speak("Muted.")


def unmute():
    run("pactl set-sink-mute @DEFAULT_SINK@ 0")
    speak("Unmuted.")


def toggle_mute():
    run("pactl set-sink-mute @DEFAULT_SINK@ toggle")
    speak("Toggled mute.")


# ---------- Time & date ----------
def tell_time():
    now = datetime.now()
    speak(now.strftime("It's %I:%M %p."))


def tell_date():
    now = datetime.now()
    speak(now.strftime("Today is %A, %B %d, %Y."))


def tell_datetime():
    now = datetime.now()
    speak(now.strftime("It's %I:%M %p on %A, %B %d."))


# ---------- Multi-step macros ----------
def run_macro(name: str):
    """Fuzzy-match a macro name and dispatch each sub-command with a small delay."""
    match = fuzzproc.extractOne(name, MACROS.keys(), scorer=fuzz.token_sort_ratio)
    if not match or match[1] < FUZZY_THRESHOLD:
        notify(f"Unknown macro: {name}")
        return
    macro_name = match[0]
    steps = MACROS[macro_name]
    speak(f"Starting {macro_name}.")

    def _dispatch():
        for step in steps:
            # Synthesise a fake "friday <step>" utterance so handle() processes it
            fake = f"{WAKE_WORD} {step}"
            handle(fake)
            time.sleep(1.2)  # small gap so windows don't race each other

    threading.Thread(target=_dispatch, daemon=True).start()


# ---------- Intent matching ----------
INTENTS = [
    # ---- Volume ----
    (r"set volume (?:to )?(\d+)", lambda m: set_volume(int(m[1]))),
    (r"volume (?:to )?(\d+)(?:\s*percent)?", lambda m: set_volume(int(m[1]))),
    (
        r"(?:turn|crank|bring) (?:the )?volume up(?: by (\d+))?",
        lambda m: change_volume(int(m[1]) if m[1] else 10),
    ),
    (
        r"(?:turn|bring) (?:the )?volume down(?: by (\d+))?",
        lambda m: change_volume(-(int(m[1]) if m[1] else 10)),
    ),
    (r"toggle mute", lambda m: toggle_mute()),
    (r"unmute", lambda m: unmute()),
    (r"mute", lambda m: mute()),
    # ---- Time & date ----
    (r"what(?:'s| is) (?:the )?time", lambda m: tell_time()),
    (r"what time is it", lambda m: tell_time()),
    (r"what(?:'s| is) (?:the )?date", lambda m: tell_date()),
    (r"what(?:'s| is) today", lambda m: tell_date()),
    (r"what(?:'s| is) (?:the )?day", lambda m: tell_date()),
    (r"date and time|time and date", lambda m: tell_datetime()),
    # ---- Multi-step macros (must come before open/launch to avoid partial match) ----
    (r"^(start my day|work mode|chill mode)$", lambda m: run_macro(m[1])),
    (
        r"(?:run|execute|start) (?:macro |mode )(.+)",
        lambda m: (
            run_macro(m[1])
            if fuzzproc.extractOne(m[1], MACROS.keys(), scorer=fuzz.token_sort_ratio)
            and fuzzproc.extractOne(m[1], MACROS.keys(), scorer=fuzz.token_sort_ratio)[
                1
            ]
            >= FUZZY_THRESHOLD
            else None
        ),
    ),
    # ---- Existing intents ----
    (
        r"open (?:my )?github (?:repositories|repos)",
        lambda m: run(
            [BROWSER, f"https://github.com/{GITHUB_USERNAME}?tab=repositories"]
        ),
    ),
    (
        r"open (?:my )?github(?: profile)?",
        lambda m: run([BROWSER, f"https://github.com/{GITHUB_USERNAME}"]),
    ),
    (
        r"open whatsapp",
        lambda m: run([BROWSER, "https://web.whatsapp.com"]),
    ),
    (
        r"open dsa",
        lambda m: run([BROWSER, "https://leetcode.com/problems/"]),
    ),
    (r"open (.+?) in zed", lambda m: run(["zed", expand(m[1])])),
    (
        r"open (.+?) in terminal",
        lambda m: run([TERMINAL, f"--working-directory={expand(m[1])}"]),
    ),
    (r"open (.+?) in files", lambda m: run(["xdg-open", expand(m[1])])),
    (r"search (?:for )?(.+?) (?:on|in) chrome", lambda m: web_search("chrome", m[1])),
    (r"search (?:for )?(.+?) (?:on|in) youtube", lambda m: web_search("youtube", m[1])),
    (
        r"(?:open (?:chat ?gpt|gpt) and )?(?:ask|search) (?:chat ?gpt|gpt) (.+)",
        lambda m: web_search("chatgpt", m[1]),
    ),
    (r"(?:system )?status report|system status", lambda m: system_report()),
    (r"(?:focus|switch to|go to) (.+)", lambda m: focus_or_launch(m[1])),
    (r"(?:open|launch|start) (\w+)$", lambda m: focus_or_launch(m[1])),
]


def handle(text):
    global asleep
    text = fuzzy_correct(text.lower().strip(" .,!?"))
    if not has_wake_word(text):
        return
    cmd = strip_wake_word(text).strip(" .,")
    if not cmd:
        return

    # sleep / wake handling
    if asleep:
        if "wake" in cmd:
            asleep = False
            beep(1100)
            speak("I'm awake.")
        return
    if re.search(r"go to sleep|sleep now|stop listening", cmd):
        asleep = True
        beep(440)
        speak("Going to sleep.")
        return

    print(f">> {cmd}", flush=True)
    beep()  # confirmation: command heard
    try:
        for pattern, action in INTENTS:
            if m := re.search(pattern, cmd):
                action(m)
                return
        # fuzzy match against docker commands first
        match = fuzzproc.extractOne(
            cmd, DOCKER_COMMANDS.keys(), scorer=fuzz.token_sort_ratio
        )
        if match and match[1] >= FUZZY_THRESHOLD:
            if run_docker_command(match[0]):
                return
        # fuzzy match against shortcut phrases
        match = fuzzproc.extractOne(cmd, SHORTCUTS.keys(), scorer=fuzz.token_sort_ratio)
        if match and match[1] >= FUZZY_THRESHOLD:
            run(SHORTCUTS[match[0]])
            notify(f"Ran: {match[0]}")
            return
        notify(f"Unknown: {cmd}")
    except FileNotFoundError as e:
        notify(f"App not found: {e.filename}")
    except Exception as e:
        notify(f"Error: {e}")


# ---------- Audio loop (VAD-gated) ----------
def main():
    print(f"Loading {MODEL_SIZE} ({DEVICE}/{COMPUTE_TYPE})...", flush=True)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")

    vad = webrtcvad.Vad(VAD_LEVEL)
    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
    audio_q = queue.Queue()

    def cb(indata, frames, t, status):
        audio_q.put(bytes(indata))

    ring = collections.deque(maxlen=int(300 / FRAME_MS))
    voiced, silence_frames = [], 0
    silence_limit = int(SILENCE_END * 1000 / FRAME_MS)
    max_frames = int(MAX_UTTERANCE * 1000 / FRAME_MS)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=frame_len,
        dtype="int16",
        channels=1,
        callback=cb,
    ):
        print(f"Listening. Say '{WAKE_WORD} <command>'", flush=True)
        while True:
            frame = audio_q.get()
            if time.time() < suppress_until:  # ignore own TTS voice
                voiced, silence_frames = [], 0
                ring.clear()
                continue
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            if not voiced:
                ring.append(frame)
                if is_speech:
                    voiced = list(ring)
                    ring.clear()
            else:
                voiced.append(frame)
                silence_frames = 0 if is_speech else silence_frames + 1
                if silence_frames >= silence_limit or len(voiced) >= max_frames:
                    audio = (
                        np.frombuffer(b"".join(voiced), dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )
                    voiced, silence_frames = [], 0
                    if len(audio) < SAMPLE_RATE * 0.4:
                        continue
                    segments, _ = model.transcribe(
                        audio,
                        language="en",
                        beam_size=1,
                        condition_on_previous_text=False,
                        vad_filter=False,
                    )
                    text = " ".join(s.text for s in segments)
                    if text.strip():
                        handle(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
