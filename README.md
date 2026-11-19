# ClipForge: viral clips from long videos

Paste a YouTube link or upload a video. AI picks the best moments and outputs captioned 9:16 clips with a face-tracking crop.

**Stack (free tier):** FastAPI + ffmpeg + OpenCV, Groq Whisper (transcription) + Groq Llama (clip picking), Docker on Render.

## Environment variables
| Key | Required | Default | Notes |
|---|---|---|---|
| GROQ_API_KEY | yes | - | https://console.groq.com/keys |
| OUT_SIZE | no | 1080x1920 | Use 720x1280 on small/free servers |
| DL_MAX_HEIGHT | no | 1080 | Max YouTube download height |
| MAX_SOURCE_MINUTES | no | 90 | Longer videos are rejected |
| MAX_UPLOAD_MB | no | 1024 | |
| FFMPEG_THREADS | no | 0 (auto) | |
| YTDLP_COOKIES_FILE / YTDLP_PROXY | no | - | Improve YouTube link success on cloud hosts |

## Run locally
```
pip install -r requirements.txt   # also needs ffmpeg
export GROQ_API_KEY=...
uvicorn app:app --port 8000
```

## Free-tier limits
- YouTube often blocks downloads from cloud servers. Users are told to use **Upload video** instead.
- Render free: 0.1 CPU / 512 MB, sleeps after 15 min idle (about 1 min cold start), local files reset on restart.
- Groq free tier rate limits: the app waits and retries.
