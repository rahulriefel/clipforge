"""Core clipping pipeline: source video -> transcript -> viral moments -> vertical captioned clips."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import yt_dlp
from groq import Groq

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq (free) | anthropic (paid)
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "16000"))  # ~4k tokens per request, fits free limits
YT_COOKIES = os.getenv("YTDLP_COOKIES_FILE")  # optional cookies.txt helps against YouTube blocks
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
PROXY_URL = os.getenv("YTDLP_PROXY")  # residential proxy, e.g. http://user:pass@host:port
MAX_SOURCE_MINUTES = int(os.getenv("MAX_SOURCE_MINUTES", "90"))
OUT_W, OUT_H = (int(x) for x in os.getenv("OUT_SIZE", "1080x1920").split("x"))
DL_MAX_HEIGHT = int(os.getenv("DL_MAX_HEIGHT", "1080"))
FF_THREADS = os.getenv("FFMPEG_THREADS", "0")  # 0 = auto


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


# ---------- 1. Get the source video ----------
def download_youtube(url: str, out_dir: Path) -> Path:
    opts = {
        "format": f"bv*[height<={DL_MAX_HEIGHT}][ext=mp4]+ba[ext=m4a]/b[height<={DL_MAX_HEIGHT}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {MAX_SOURCE_MINUTES * 60}"),
    }
    if PROXY_URL:
        opts["proxy"] = PROXY_URL
    if YT_COOKIES and Path(YT_COOKIES).exists():
        opts["cookiefile"] = YT_COOKIES
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        if "Sign in" in str(e) or "bot" in str(e).lower() or "403" in str(e):
            raise RuntimeError("YouTube blocked the download from our server. "
                               "Please download the video and use 'Upload video' instead.") from e
        raise
    path = out_dir / "source.mp4"
    if not path.exists():
        raise RuntimeError("Download failed or video longer than allowed limit")
    return path


# ---------- 2. Transcribe ----------
def extract_audio(video: Path, out_dir: Path) -> Path:
    audio = out_dir / "audio.mp3"
    # mono 16 kHz, low bitrate keeps file under Groq's 25 MB limit for ~90 min
    try:
        run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)])
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Could not read audio from this video. Clips are picked from speech, "
                           "so the video needs a voice track.") from e
    return audio


def transcribe(audio: Path) -> list[dict]:
    client = Groq()
    with open(audio, "rb") as f:
        res = client.audio.transcriptions.create(
            file=(audio.name, f.read()),
            model=WHISPER_MODEL,
            response_format="verbose_json",
        )
    segs = res.segments if hasattr(res, "segments") else res["segments"]
    return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()} for s in segs]


# ---------- 3. Find viral moments ----------
PICK_PROMPT = """You are a short-form video editor who finds viral moments for TikTok, Reels and Shorts.
Below is a timestamped transcript. Pick the {n} best standalone clips.

Rules:
- Each clip 20-60 seconds, starts on a strong hook, ends on a complete thought.
- Use transcript timestamps exactly (start of first segment, end of last segment).
- Clips must not overlap.
- Score virality 0-100 (hook strength, emotion, surprise, value, shareability).

Return ONLY JSON: {{"clips":[{{"start":float,"end":float,"title":str,"hook":str,"score":int,"reason":str}}]}}

Transcript:
{transcript}"""


GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant"]
_working_model: str | None = None


def _ask_groq(prompt: str) -> str:
    """Try the configured model, then fallbacks; Groq retires models often. Remembers what works."""
    global _working_model
    models = [m for m in dict.fromkeys([_working_model, GROQ_LLM_MODEL, *GROQ_FALLBACK_MODELS]) if m]
    last = None
    for model in models:
        for json_mode in (True, False):
            try:
                kw = {"response_format": {"type": "json_object"}} if json_mode else {}
                res = Groq().chat.completions.create(
                    model=model, temperature=0.3, max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}], **kw)
                _working_model = model
                return res.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "rate_limit" in msg or "rate limit" in msg.lower():
                    raise
                last = e
                if "model_not_found" in msg or "does not exist" in msg or "decommissioned" in msg:
                    break  # try next model
                if "response_format" in msg or "json" in msg.lower():
                    continue  # retry same model without JSON mode
                break
    raise RuntimeError(f"No Groq model available: {last}")


def _ask_llm(prompt: str) -> str:
    """Send prompt to the configured LLM (free default: Groq). Retries on rate limits."""
    for attempt in range(5):
        try:
            if LLM_PROVIDER == "anthropic":
                from anthropic import Anthropic
                msg = Anthropic().messages.create(model=CLAUDE_MODEL, max_tokens=2000,
                                                  messages=[{"role": "user", "content": prompt}])
                return msg.content[0].text
            return _ask_groq(prompt)
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "rate_limit" in str(e) or "rate limit" in str(e).lower():
                time.sleep(20 * (attempt + 1))  # free-tier tokens-per-minute limit
                continue
            raise
    raise RuntimeError("AI rate limit reached, try again in a few minutes")


def _chunks(segments: list[dict], max_chars: int):
    cur, size = [], 0
    for s in segments:
        line = f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        if cur and size + len(line) > max_chars:
            yield cur
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        yield cur


def pick_clips(segments: list[dict], n: int = 5) -> list[dict]:
    # Free-tier LLMs have small per-minute token limits, so long transcripts are scanned in windows.
    windows = list(_chunks(segments, CHUNK_CHARS))
    per_window = max(2, -(-n * 2 // len(windows)))  # ask for extra candidates, keep the best n
    found = []
    for lines in windows:
        text = _ask_llm(PICK_PROMPT.format(n=per_window, transcript="\n".join(lines)))
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            continue
        try:
            found += json.loads(m.group(0)).get("clips", [])
        except json.JSONDecodeError:
            continue
    valid = []
    for c in sorted(found, key=lambda c: -int(c.get("score", 0))):
        try:
            c["start"], c["end"] = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 10 <= c["end"] - c["start"] <= 90:
            continue
        if any(c["start"] < v["end"] and c["end"] > v["start"] for v in valid):
            continue
        valid.append(c)
        if len(valid) == n:
            break
    if not valid:
        raise RuntimeError("Could not find clip-worthy moments in this video")
    return valid


# ---------- 4. Cut, reframe 9:16, burn captions ----------
def _ts(t: float) -> str:
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


def write_srt(segments: list[dict], start: float, end: float, path: Path) -> None:
    lines, i = [], 1
    for s in segments:
        if s["end"] <= start or s["start"] >= end:
            continue
        words = s["text"].split()
        a, b = max(s["start"], start) - start, min(s["end"], end) - start
        # split long segments into ~4-word caption chunks
        chunks = [words[k:k + 4] for k in range(0, len(words), 4)] or [[]]
        step = (b - a) / len(chunks)
        for j, chunk in enumerate(chunks):
            lines.append(f"{i}\n{_ts(a + j * step)} --> {_ts(a + (j + 1) * step)}\n{' '.join(chunk).upper()}\n")
            i += 1
    path.write_text("\n".join(lines), encoding="utf-8")


_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _frame_plan(video: Path, start: float, end: float) -> str:
    """Return the ffmpeg filter that makes a 1080x1920 frame, crop centred on the speaker's face."""
    cap = cv2.VideoCapture(str(video))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not w or not h or w / h <= 9 / 16 + 0.01:  # already vertical: fit and pad
        cap.release()
        return f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2"
    centers, t = [], start
    step = max(1.0, (end - start) / 30)  # at most ~30 samples per clip
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        t += step
        if not ok:
            continue
        small = cv2.resize(frame, (960, int(960 * h / w)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = _FACE.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
        if len(faces):
            x, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])
            centers.append((x + fw / 2) / 960)
    cap.release()
    cx = sorted(centers)[len(centers) // 2] if centers else 0.5  # median face position
    crop_w = int(h * 9 / 16) // 2 * 2
    x = int(min(max(cx * w - crop_w / 2, 0), w - crop_w))
    return f"crop={crop_w}:{int(h)//2*2}:{x}:0,scale={OUT_W}:{OUT_H}"


def render_clip(video: Path, segments: list[dict], clip: dict, out: Path) -> None:
    srt = out.with_suffix(".srt")
    write_srt(segments, clip["start"], clip["end"], srt)
    style = ("FontName=Arial,FontSize=14,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV=60")
    vf = (
        _frame_plan(video, clip["start"], clip["end"]) + ","
        f"subtitles={srt.as_posix()}:force_style='{style}'"
    )
    run([
        "ffmpeg", "-y", "-ss", f"{clip['start']:.2f}", "-to", f"{clip['end']:.2f}", "-i", str(video),
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", FF_THREADS,
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ])
    srt.unlink(missing_ok=True)


# ---------- Orchestrator ----------
def process(job_dir: Path, url: str | None = None, uploaded: Path | None = None,
            n_clips: int = 5, on_status=lambda s: None) -> list[dict]:
    job_dir.mkdir(parents=True, exist_ok=True)
    on_status("downloading")
    video = download_youtube(url, job_dir) if url else uploaded
    on_status("transcribing")
    segments = transcribe(extract_audio(video, job_dir))
    on_status("finding viral moments")
    clips = pick_clips(segments, n_clips)
    results = []
    for i, c in enumerate(clips, 1):
        on_status(f"rendering clip {i}/{len(clips)}")
        out = job_dir / f"clip_{len(results) + 1}.mp4"
        try:
            render_clip(video, segments, c, out)
        except subprocess.CalledProcessError:
            continue  # skip a bad clip instead of failing the whole job
        if out.exists() and out.stat().st_size > 10_000:
            results.append({**c, "file": out.name})
    if not results:
        raise RuntimeError("Could not render any clips from this video")
    on_status("done")
    return results
