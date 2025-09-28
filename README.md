# YT-Dash

Lightweight Gradio app to download YouTube videos/audio and transcribe them using **faster-whisper**.

---

## Summary

**YT-Dash** is a small web UI (Gradio) that lets you:
- Inspect available formats for a YouTube video (audio & video streams)
- Download a selected stream (or full-resolution merged video)
- Transcribe audio using **faster-whisper** (with optional chunking / VAD fallback)
- Produce outputs: plain transcript, timestamped transcript, SRT subtitles and translations (English via Whisper `translate`; Hindi via `google_trans_new` if installed)

All downloads & generated files are written to the `downloads/` folder.

---

## Features

- Automatic format discovery via `yt-dlp`
- Download audio-only or video streams, or `bestvideo+bestaudio` merged
- Transcription with `faster-whisper` and fallback to chunked mode when memory/timeout occurs
- Transcript outputs: `.transcript.txt`, `.transcript_timestamped.txt`, `.transcript.srt`
- English translation via Whisper `translate` task (if model supports it)
- Hindi translation via `google_trans_new` if available
- Progress updates in the Gradio UI and graceful error handling

---

## Requirements

- Python 3.8+
- `ffmpeg` installed and accessible (or set `FFMPEG_PATH` variable in the script)
- Recommended Python packages (example `requirements.txt` below):

```
gradio
yt-dlp
faster-whisper
pydub
google_trans_new
```

> Note: `google_trans_new` is optional — if missing, Hindi translation will show a helpful message.

---

## Quick start

1. Create & activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `ffmpeg` is installed. If it's not in your PATH, set `FFMPEG_PATH` in the script to the `bin` folder of ffmpeg.

4. Run the app:

```bash
python ytdash.py
```

Replace `ytdash.py` with the actual filename of the script if different.

Open the Gradio URL printed in the terminal (usually `http://127.0.0.1:7860`).


---
## Configuration (script-level variables)

At the top of the script you can change:

```python
FFMPEG_PATH = r"D:\\ffmpeg-...\\bin"        # optional, set to your ffmpeg folder
MODEL_SIZE = "medium"                       # whisper model size: small, medium, large, etc.
DEVICE = "cuda"                             # or "cpu"
COMPUTE_TYPE = "float16"                    # or "int8"/"float32" depending on model/device
BEAM_SIZE = 5
VAD_FILTER = False                            # enable/disable VAD filtering
FORCE_LANGUAGE = "en"                       # force transcription language (optional)
CHUNK_SECONDS = 60                            # chunk size in seconds for fallback
CHUNK_OVERLAP_SECONDS = 1.0                   # overlap between chunks
```

Tweak these values to match your machine (GPU memory, CPU-only, etc.).

---

## Usage notes

1. Paste a YouTube URL into the input box and click **Submit**.
2. The UI will display a thumbnail, title, and available audio/video formats.
3. Click **Audio** / **Video** / **Text** to populate the quality dropdown.
4. Choose a quality and click **Download / Show** to save the file to `downloads/`.
5. To transcribe, choose a format (or **Use best audio (automatic)**) and click **Download + Transcribe (Text)**.
6. After transcription completes, generated files will appear in `downloads/` and the file pickers will show links.

---

## Output files

The script writes the following files to `downloads/` (prefix = video title or filename):

- `{prefix}.transcript.txt` — Plain transcript
- `{prefix}.transcript_timestamped.txt` — Timestamped transcript
- `{prefix}.transcript.srt` — SRT subtitle file
- `{prefix}.translation_en.txt` — English translation (Whisper `translate`)
- `{prefix}.translation_en_timestamped.txt`
- `{prefix}.translation_en.srt`
- `{prefix}.translation_hi.txt` — Hindi translation (if available)

---

## Troubleshooting

- **faster-whisper import fails**: Install with `pip install faster-whisper` and make sure your CUDA/cuDNN and PyTorch are compatible if using GPU.
- **ffmpeg errors**: Ensure ffmpeg is installed and `FFMPEG_PATH` is set if ffmpeg is not in your PATH.
- **pydub errors**: `pydub` requires ffmpeg; install `pydub` via pip and confirm ffmpeg path.
- **`google_trans_new` failing/limited**: This package sometimes breaks due to Google changes — treat Hindi translation as optional.
- **Out of memory during transcription**: Lower `MODEL_SIZE`, set `DEVICE = \"cpu\"` or use the chunked transcription fallback (`CHUNK_SECONDS`).

If issues persist, open an issue with the script's traceback and environment details (OS, Python version, GPU model).

---

## Security & Privacy

- This script downloads content from YouTube — ensure you have the right to download and transcribe the content.
- Transcription runs locally on your machine (unless you run it on a remote server) — files are stored locally under `downloads/`.

---

## Suggested `requirements.txt`

```
gradio>=3.0
yt-dlp
faster-whisper
pydub
google_trans_new
```

Add or pin versions as needed for your environment.

---

## How to publish to GitHub (quick commands)

```bash
git init
git add .
git commit -m "Initial commit: add YT-Dash Gradio app"
# create repo on GitHub (or use GitHub CLI): gh repo create <repo-name> --public --source=. --push
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

---

## Contributing

Contributions welcome — forks, PRs and issues. Please add clear PR descriptions and small focused commits.

---

## License

MIT License — see `LICENSE` (or add your preferred license).

---

## Author

Your Name — replace this with your name and contact link (e.g., GitHub profile, email).

---

*Generated README for the `YT-Dash` Gradio app.*
