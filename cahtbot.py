import gradio as gr
import yt_dlp
import json
import os
import glob
import time
import warnings
import math
import concurrent.futures
import traceback

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from google_trans_new import google_translator
    GOOGLETRANS_AVAILABLE = True
except Exception:
    GOOGLETRANS_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

FFMPEG_PATH = r"D:\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
VAD_FILTER = False
FORCE_LANGUAGE = "en"
CHUNK_SECONDS = 60
CHUNK_OVERLAP_SECONDS = 1.0

os.makedirs("downloads", exist_ok=True)

SPINNER_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;gap:8px;padding:12px;">
  <div class="lds-ring"><div></div><div></div><div></div><div></div></div>
  <div style="font-size:0.95em;color:#444;">Working... please wait</div>
</div>
<style>
.lds-ring{display:inline-block;position:relative;width:48px;height:48px}
.lds-ring div{box-sizing:border-box;display:block;position:absolute;width:40px;height:40px;margin:4px;border:4px solid #ff8c00;border-radius:50%;animation:lds-ring 1.2s cubic-bezier(.5,0,.5,1) infinite;border-color:#ff8c00 transparent transparent transparent}
@keyframes lds-ring{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
</style>
"""

def human_size(num):
    if not num:
        return "unknown"
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"

def seconds_to_timestamp(s):
    ms = int((s - math.floor(s)) * 1000)
    s_int = int(math.floor(s))
    h = s_int // 3600
    m = (s_int % 3600) // 60
    sec = s_int % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

def seconds_to_srt_timestamp(s):
    ms = int(round((s - math.floor(s)) * 1000))
    if ms >= 1000:
        ms = 999
    s_int = int(math.floor(s))
    h = s_int // 3600
    m = (s_int % 3600) // 60
    sec = s_int % 60
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def segments_to_timestamped_text(segments, prefix=""):
    lines = []
    for seg in segments:
        start = getattr(seg, "start", None)
        end = getattr(seg, "end", None)
        start_ts = seconds_to_timestamp(start) if start is not None else ""
        end_ts = seconds_to_timestamp(end) if end is not None else ""
        text = getattr(seg, "text", "").strip()
        if text:
            lines.append(f"{prefix}[{start_ts} --> {end_ts}] {text}")
    return "\n".join(lines)

def chunk_segments_to_text(segments):
    parts = []
    for seg in segments:
        t = getattr(seg, "text", "").strip()
        if t:
            parts.append(t)
    return " ".join(parts)

def write_srt_from_segments(segments, outpath):
    """
    Write SRT file from whisper segments (expects start/end/text on each segment).
    """
    try:
        with open(outpath, "w", encoding="utf-8") as f:
            idx = 1
            for seg in segments:
                text = getattr(seg, "text", "").strip()
                if not text:
                    continue
                start = getattr(seg, "start", 0.0)
                end = getattr(seg, "end", 0.0)
                f.write(f"{idx}\n")
                f.write(f"{seconds_to_srt_timestamp(float(start))} --> {seconds_to_srt_timestamp(float(end))}\n")
                f.write(text.replace("\n", " ").strip() + "\n\n")
                idx += 1
        return True
    except Exception as e:
        warnings.warn(f"Failed to write SRT {outpath}: {e}")
        return False

_model_cache = None

def get_whisper_model(progress_cb=None):
    """Return a cached WhisperModel instance (loads once). Reports via progress_cb if provided."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if callable(progress_cb):
        try:
            progress_cb("Loading Whisper model...")
        except Exception:
            pass
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed or failed import. Install with: pip install faster-whisper")
    try:
        _model_cache = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        warnings.warn(f"Model load failed with device={DEVICE}, compute_type={COMPUTE_TYPE}: {e}\nFalling back to cpu.")
        _model_cache = WhisperModel(MODEL_SIZE, device="cpu")
    return _model_cache

def robust_transcribe_call(model, audio_path, task="transcribe", language=None, beam_size=1, vad_filter=False, vad_params=None, offset=0.0):
    base_kwargs = dict(task=task, beam_size=beam_size)
    if language:
        base_kwargs["language"] = language

    if vad_filter:
        try_kwargs = dict(**base_kwargs, vad_filter=True, vad_parameters=vad_params if vad_params else None)
    else:
        try_kwargs = dict(**base_kwargs, vad_filter=False)

    try:
        segments, info = model.transcribe(audio_path, **try_kwargs)
    except TypeError:
        try:
            segments, info = model.transcribe(audio_path, task=task, beam_size=beam_size, vad_filter=vad_filter, language=language)
        except TypeError:
            segments, info = model.transcribe(audio_path, task=task, beam_size=beam_size, language=language)

    if offset and segments:
        for seg in segments:
            if hasattr(seg, "start"):
                seg.start = float(seg.start) + offset
            if hasattr(seg, "end"):
                seg.end = float(seg.end) + offset
    return segments, info

def transcribe_by_chunks(model, audio_path, task="transcribe", language=None, beam_size=1, vad_filter=False, vad_params=None, chunk_seconds=60, overlap_seconds=1.0, progress_cb=None):
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub not installed. Install with: pip install pydub (and ffmpeg).")
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = int(chunk_seconds * 1000)
    overlap_ms = int(overlap_seconds * 1000)
    combined_segments = []
    last_info = None
    start_ms = 0
    chunk_index = 0
    total_chunks = max(1, math.ceil((total_ms - overlap_ms) / (chunk_ms - overlap_ms)))
    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_ms, total_ms)
        chunk = audio[start_ms:end_ms]
        chunk_filename = os.path.join("downloads", f"__tmp_chunk_{int(time.time())}_{chunk_index}.wav")
        chunk.export(chunk_filename, format="wav")
        chunk_start_sec = start_ms / 1000.0
        if callable(progress_cb):
            try:
                progress_cb(f"Processing chunk {chunk_index+1}/{total_chunks} (offset {chunk_start_sec:.1f}s)...")
            except Exception:
                pass
        segments, info = robust_transcribe_call(
            model,
            chunk_filename,
            task=task,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_params=vad_params,
            offset=chunk_start_sec
        )
        if segments:
            combined_segments.extend(segments)
        last_info = info
        try:
            os.remove(chunk_filename)
        except Exception:
            pass
        start_ms = end_ms - overlap_ms
        chunk_index += 1
    return combined_segments, last_info

def run_pipeline_on_file(file_path, prefix=None, progress_cb=None):
    """
    Runs transcription + translation on a local file, saves outputs to downloads/ and returns paths.
    Uses a simple model.transcribe(...) call first (fast), falls back to robust chunking if needed.
    progress_cb: optional callable(msg) to report progress (used by background thread).
    Returns dict with keys: plain, timestamped, srt, en, en_ts, en_srt, hi, seconds
    """
    start_total = time.time()
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(file_path))[0]

    if callable(progress_cb):
        try:
            progress_cb("Preparing model...")
        except Exception:
            pass

    model = get_whisper_model(progress_cb=progress_cb)

    try:
        if callable(progress_cb):
            try:
                progress_cb("Running fast full-file transcription (transcribe)...")
            except Exception:
                pass
        segments_orig, info_orig = robust_transcribe_call(
            model,
            file_path,
            task="transcribe",
            language=FORCE_LANGUAGE if FORCE_LANGUAGE else None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            vad_params=None
        )
    except (RuntimeError, MemoryError, Exception) as e:
        if callable(progress_cb):
            try:
                progress_cb(f"Full-file transcription failed ({type(e).__name__}), falling back to chunked mode...")
            except Exception:
                pass
        segments_orig, info_orig = transcribe_by_chunks(
            model,
            file_path,
            task="transcribe",
            language=FORCE_LANGUAGE if FORCE_LANGUAGE else None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            vad_params=None,
            chunk_seconds=CHUNK_SECONDS,
            overlap_seconds=CHUNK_OVERLAP_SECONDS,
            progress_cb=progress_cb
        )

    plain_transcript_file = os.path.join("downloads", prefix + ".transcript.txt")
    ts_transcript_file = os.path.join("downloads", prefix + ".transcript_timestamped.txt")
    srt_file = os.path.join("downloads", prefix + ".transcript.srt")
    try:
        with open(plain_transcript_file, "w", encoding="utf-8") as f:
            f.write(chunk_segments_to_text(segments_orig))
        with open(ts_transcript_file, "w", encoding="utf-8") as f:
            f.write(segments_to_timestamped_text(segments_orig))
        write_srt_from_segments(segments_orig, srt_file)
    except Exception as e:
        warnings.warn(f"Failed to write original transcript files: {e}")

    # English translation via Whisper's translate task
    try:
        if callable(progress_cb):
            try:
                progress_cb("Running translate task (to English)...")
            except Exception:
                pass
        segments_en, info_en = robust_transcribe_call(
            model,
            file_path,
            task="translate",
            language=None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            vad_params=None
        )
    except (RuntimeError, MemoryError, Exception) as e:
        if callable(progress_cb):
            try:
                progress_cb(f"Translate task failed ({type(e).__name__}), falling back to chunked translate...")
            except Exception:
                pass
        segments_en, info_en = transcribe_by_chunks(
            model,
            file_path,
            task="translate",
            language=None,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            vad_params=None,
            chunk_seconds=CHUNK_SECONDS,
            overlap_seconds=CHUNK_OVERLAP_SECONDS,
            progress_cb=progress_cb
        )

    translation_en_file = os.path.join("downloads", prefix + ".translation_en.txt")
    translation_en_ts_file = os.path.join("downloads", prefix + ".translation_en_timestamped.txt")
    translation_en_srt_file = os.path.join("downloads", prefix + ".translation_en.srt")
    try:
        with open(translation_en_file, "w", encoding="utf-8") as f:
            f.write(chunk_segments_to_text(segments_en))
        with open(translation_en_ts_file, "w", encoding="utf-8") as f:
            f.write(segments_to_timestamped_text(segments_en))
        write_srt_from_segments(segments_en, translation_en_srt_file)
    except Exception as e:
        warnings.warn(f"Failed to write English translation files: {e}")

    translation_hi_file = os.path.join("downloads", prefix + ".translation_hi.txt")
    if GOOGLETRANS_AVAILABLE:
        try:
            if callable(progress_cb):
                try:
                    progress_cb("Translating to Hindi via google_trans_new...")
                except Exception:
                    pass
            translator = google_translator()
            source_text = chunk_segments_to_text(segments_en) or chunk_segments_to_text(segments_orig)
            CHUNK = 4000
            translated_parts = []
            for i in range(0, len(source_text), CHUNK):
                piece = source_text[i:i+CHUNK]
                translated = translator.translate(piece, lang_tgt='hi')
                translated_parts.append(translated)
            translation_hi = "\n".join(translated_parts)
            with open(translation_hi_file, "w", encoding="utf-8") as f:
                f.write(translation_hi)
        except Exception as e:
            with open(translation_hi_file, "w", encoding="utf-8") as f:
                f.write("Hindi translation failed: " + str(e))
    else:
        with open(translation_hi_file, "w", encoding="utf-8") as f:
            f.write("google_trans_new not installed. To enable Hindi translation run: pip install google_trans_new\n")

    total_time = time.time() - start_total
    results = {
        "plain": plain_transcript_file,
        "timestamped": ts_transcript_file,
        "srt": srt_file,
        "en": translation_en_file,
        "en_ts": translation_en_ts_file,
        "en_srt": translation_en_srt_file,
        "hi": translation_hi_file,
        "seconds": total_time
    }

    if callable(progress_cb):
        try:
            progress_cb(f"Completed in {total_time:.1f}s")
        except Exception:
            pass

    return results


def handle_input(url_input):
    if not url_input:
        return None, "Please paste a YouTube URL.", "", ""
    try:
        ydl_opts = {'quiet': True}
        if FFMPEG_PATH:
            ydl_opts['ffmpeg_location'] = FFMPEG_PATH

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_input, download=False)
            title = info.get('title') or "No title"
            video_id = info.get('id')
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        video_url = f"https://youtu.be/{video_id}"

        video_formats = []
        audio_formats = []
        seen_video_ids = set()
        seen_audio_ids = set()

        for f in info.get('formats', []):
            fmt_id = f.get('format_id')
            if not fmt_id:
                continue

            filesize = f.get('filesize') or f.get('filesize_approx') or None
            size_str = human_size(filesize) if filesize else "unknown"
            ext = f.get('ext') or ''
            has_audio = f.get('acodec') not in (None, 'none')
            if f.get('vcodec') in (None, 'none') and f.get('acodec') not in (None, 'none'):
                if fmt_id in seen_audio_ids:
                    continue
                seen_audio_ids.add(fmt_id)
                abr = f.get('abr') or f.get('tbr') or "unknown"
                label = f"{abr} kbps · {ext} · {size_str} · id:{fmt_id}"
                audio_formats.append({
                    'label': label, 'format_id': fmt_id,
                    'filesize': filesize, 'has_audio': True, 'ext': ext, 'type': 'audio'
                })

            # Video streams
            elif f.get('vcodec') not in (None, 'none'):
                if fmt_id in seen_video_ids:
                    continue
                seen_video_ids.add(fmt_id)
                height = f.get('height') or f.get('format_note') or "unknown"
                has_audio_text = 'with audio' if has_audio else 'video-only'
                label = f"{height}p · {ext} · {has_audio_text} · {size_str} · id:{fmt_id}"
                video_formats.append({
                    'label': label, 'format_id': fmt_id,
                    'filesize': filesize, 'has_audio': has_audio, 'ext': ext, 'type': 'video'
                })

        formats_json = json.dumps({'video': video_formats, 'audio': audio_formats})
        return thumbnail_url, title, video_url, formats_json
    except Exception as e:
        return None, f"Error: {str(e)}", "", ""

def submit_action(url_input):
    thumb, title, video_url, formats_json = handle_input(url_input)
    if thumb is None:
        return None, f"**{title}**", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)
    return thumb, f"**{title}**", formats_json, video_url, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), "", gr.update(visible=False), gr.update(visible=False)

def populate_quality_dropdown(mode, formats_json):
    try:
        data = json.loads(formats_json or "{}")
        items = data.get(mode, [])
        labels = [it['label'] for it in items]
        if mode == 'video':
            labels.insert(0, "🔝 Full resolution (bestvideo+bestaudio)")
        if not labels:
            return gr.update(choices=[], value=None, visible=True), gr.update(visible=False), "<i>No available formats for this type.</i>"
        return gr.update(choices=labels, value=labels[0], visible=True), gr.update(visible=True), ""
    except Exception as e:
        return gr.update(choices=[], value=None, visible=True), gr.update(visible=False), f"<i>Error listing formats: {e}</i>"

def populate_quality_for_text(formats_json):
    try:
        data = json.loads(formats_json or "{}")
        audio_items = data.get('audio', [])
        video_items = data.get('video', [])
        labels = []
        labels.append("**Use best audio (automatic)**")
        labels.extend([it['label'] for it in audio_items])
        if not audio_items:
            labels.extend([it['label'] for it in video_items])
        if not labels:
            return gr.update(choices=[], value=None, visible=True), gr.update(visible=False), "<i>No available formats for text/transcription.</i>"
        return gr.update(choices=labels, value=labels[0], visible=True), gr.update(visible=True), ""
    except Exception as e:
        return gr.update(choices=[], value=None, visible=True), gr.update(visible=False), f"<i>Error listing formats: {e}</i>"

def make_download_link(video_url, selected_label, formats_json):
    if not video_url:
        return "<i>No video URL available.</i>"
    if not selected_label:
        return "<i>Please select a quality first.</i>"
    try:
        data = json.loads(formats_json or "{}")
        found = None
        for typ in ('video','audio'):
            for it in data.get(typ, []):
                if it['label'] == selected_label:
                    found = it
                    break
            if found:
                break

        if not found:
            if selected_label == "**Use best audio (automatic)**":
                return "<p>Selected: <b>Use best audio (automatic)</b></p><p>Will download best available audio (m4a) to transcribe.</p>"
            if selected_label == "🔝 Full resolution (bestvideo+bestaudio)":
                return "<p>Selected: <b>Full resolution</b></p><p>Will download best available video + best audio and merge (requires ffmpeg).</p>"
            return "<i>Selected format not found.</i>"

        size_str = human_size(found.get('filesize')) if found.get('filesize') else "unknown"
        has_audio = found.get('has_audio')
        info_html = (f"<p>Selected: <b>{selected_label}</b></p>"
                    f"<p>Format id: <code>{found.get('format_id')}</code></p>"
                    f"<p>Estimated size: <b>{size_str}</b></p>"
                    f"<p>Has audio: <b>{'Yes' if has_audio else 'No (will merge with bestaudio)'}</b></p>"
                    "<p style='margin-top:8px;font-size:0.9em;color:#444;'>"
                    "Click <b>Download / Show</b> to download the selected format to the server. "
                    "If the selected video is video-only, the server will download best audio (m4a) and merge (requires ffmpeg)."
                    "</p>")
        return info_html
    except Exception as e:
        return f"<i>Error preparing download link: {e}</i>"

def clear_action():
    return None, "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False)

def download_selected_stream(selected_label, formats_json, video_url, progress=gr.Progress()):
    """
    Generator that shows spinner, downloads the selected format (or full-res),
    then returns file and info HTML. Outputs: (file_output, download_link_html)
    """
    if not video_url:
        yield gr.update(visible=False, value=None), "<i>No video URL available.</i>"
        return
    if not selected_label:
        yield gr.update(visible=False, value=None), "<i>Please select a quality first.</i>"
        return

    try:
        progress(0.03, desc="Preparing download...")
        yield gr.update(visible=False, value=None), SPINNER_HTML

        data = json.loads(formats_json or "{}")
        filename = None

        if selected_label == "🔝 Full resolution (bestvideo+bestaudio)":
            format_spec = "bestvideo+bestaudio/best"
            ydl_opts = {
                'format': format_spec,
                'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                'quiet': True,
                'noprogress': False,
                'merge_output_format': 'mp4',
                'noplaylist': True,
            }
            if FFMPEG_PATH:
                ydl_opts['ffmpeg_location'] = FFMPEG_PATH

            progress(0.15, desc="Downloading best video + best audio...")
            yield gr.update(visible=False, value=None), "<i>Downloading best video + best audio...</i>"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                try:
                    filename = ydl.prepare_filename(info)
                except Exception:
                    filename = None
        else:
            found = None
            for it in data.get('audio', []) + data.get('video', []):
                if it['label'] == selected_label:
                    found = it
                    break

            if not found:
                yield gr.update(visible=False, value=None), "<i>Selected format not found.</i>"
                return

            if found.get('type') == 'audio':
                ydl_opts = {
                    'format': found['format_id'],
                    'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                    'quiet': True,
                    'noprogress': False,
                    'noplaylist': True,
                }
                if FFMPEG_PATH:
                    ydl_opts['ffmpeg_location'] = FFMPEG_PATH

                progress(0.15, desc="Downloading audio...")
                yield gr.update(visible=False, value=None), f"<i>Downloading audio ({found.get('label')})...</i>"
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    try:
                        filename = ydl.prepare_filename(info)
                    except Exception:
                        filename = None
            else:
                if found.get('has_audio'):
                    format_spec = found['format_id']
                else:
                    format_spec = f"{found['format_id']}+bestaudio[ext=m4a]/best"
                ydl_opts = {
                    'format': format_spec,
                    'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                    'quiet': True,
                    'noprogress': False,
                    'merge_output_format': 'mp4',
                    'noplaylist': True,
                }
                if FFMPEG_PATH:
                    ydl_opts['ffmpeg_location'] = FFMPEG_PATH

                progress(0.15, desc="Downloading video+audio...")
                yield gr.update(visible=False, value=None), f"<i>Downloading video+audio ({found.get('label')})...</i>"
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    try:
                        filename = ydl.prepare_filename(info)
                    except Exception:
                        filename = None

        if not filename or not os.path.exists(filename):
            files = glob.glob(os.path.join("downloads", "*"))
            if not files:
                yield gr.update(visible=False, value=None), "<i>Download completed but file not found.</i>"
                return
            filename = max(files, key=os.path.getmtime)

        name = os.path.basename(filename)
        html_msg = f"<p>Downloaded: <b>{name}</b></p><p>You can click the file below to download it to your machine.</p>"
        progress(1.0, desc="Done")
        yield gr.update(value=filename, visible=True), html_msg

    except Exception as e:
        print("Error in download_selected_stream:", traceback.format_exc())
        yield gr.update(visible=False, value=None), f"<i>Error during download: {e}</i>"

def download_and_transcribe_stream(selected_label, formats_json, video_url, progress=gr.Progress()):
    """Download and transcribe with progress updates — runs transcription in a thread and yields periodic status to keep connection alive."""
    if not video_url:
        yield gr.update(visible=False, value=None), "<i>No video URL available.</i>"
        return

    if not selected_label:
        yield gr.update(visible=False, value=None), "<i>Please select a quality first.</i>"
        return

    try:
        progress(0.03, desc="Preparing download...")
        yield gr.update(visible=False, value=None), SPINNER_HTML

        data = json.loads(formats_json or "{}")
        filename = None

        if selected_label == "**Use best audio (automatic)**":
            format_spec = "bestaudio[ext=m4a]/best"
            ydl_opts = {
                'format': format_spec,
                'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                'quiet': True,
                'noprogress': False,
                'noplaylist': True,
            }
            if FFMPEG_PATH:
                ydl_opts['ffmpeg_location'] = FFMPEG_PATH

            progress(0.15, desc="Downloading best audio...")
            yield gr.update(visible=False, value=None), "<i>Downloading best audio...</i>"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                try:
                    filename = ydl.prepare_filename(info)
                except Exception:
                    filename = None
        elif selected_label == "🔝 Full resolution (bestvideo+bestaudio)":
            format_spec = "bestaudio[ext=m4a]/best"
            ydl_opts = {
                'format': format_spec,
                'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                'quiet': True,
                'noprogress': False,
                'noplaylist': True,
            }
            if FFMPEG_PATH:
                ydl_opts['ffmpeg_location'] = FFMPEG_PATH

            progress(0.15, desc="Downloading audio (for transcription)...")
            yield gr.update(visible=False, value=None), "<i>Downloading best audio for transcription...</i>"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                try:
                    filename = ydl.prepare_filename(info)
                except Exception:
                    filename = None
        else:
            # find selected format
            found = None
            for it in data.get('audio', []) + data.get('video', []):
                if it['label'] == selected_label:
                    found = it
                    break

            if not found:
                yield gr.update(visible=False, value=None), "<i>Selected format not found.</i>"
                return

            if found.get('type') == 'audio':
                ydl_opts = {
                    'format': found['format_id'],
                    'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                    'quiet': True,
                    'noprogress': False,
                    'noplaylist': True,
                }
                if FFMPEG_PATH:
                    ydl_opts['ffmpeg_location'] = FFMPEG_PATH

                progress(0.15, desc="Downloading audio...")
                yield gr.update(visible=False, value=None), f"<i>Downloading audio ({found.get('label')})...</i>"
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    try:
                        filename = ydl.prepare_filename(info)
                    except Exception:
                        filename = None
            else:
                format_spec = f"{found['format_id']}+bestaudio[ext=m4a]/best"
                ydl_opts = {
                    'format': format_spec,
                    'outtmpl': os.path.join("downloads", "%(title)s - %(id)s.%(ext)s"),
                    'quiet': True,
                    'noprogress': False,
                    'merge_output_format': 'mp4',
                    'noplaylist': True,
                }
                if FFMPEG_PATH:
                    ydl_opts['ffmpeg_location'] = FFMPEG_PATH

                progress(0.15, desc="Downloading video+audio...")
                yield gr.update(visible=False, value=None), f"<i>Downloading video+audio ({found.get('label')})...</i>"
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    try:
                        filename = ydl.prepare_filename(info)
                    except Exception:
                        filename = None

        if not filename or not os.path.exists(filename):
            files = glob.glob(os.path.join("downloads", "*"))
            if not files:
                yield gr.update(visible=False, value=None), "<i>Download completed but file not found.</i>"
                return
            filename = max(files, key=os.path.getmtime)

        base_name = os.path.splitext(os.path.basename(filename))[0]
        progress(0.25, desc="Starting transcription...")
        yield gr.update(visible=False, value=None), f"<i>Downloaded: <b>{os.path.basename(filename)}</b>. Starting transcription...</i>"

        latest = {"msg": "Transcription queued..."}

        def progress_cb(msg):
            try:
                latest["msg"] = str(msg)
            except:
                latest["msg"] = "Working..."

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_pipeline_on_file, filename, base_name, progress_cb)

            while not future.done():
                yield gr.update(visible=False, value=None), f"<i>{latest['msg']}</i>"
                time.sleep(1.0)

            try:
                results = future.result()
            except Exception as e:
                err_html = f"<i>Transcription failed: {e}</i>"
                print("Transcription background error:", traceback.format_exc())
                yield gr.update(visible=False, value=None), err_html
                return

        progress(0.9, desc="Finalizing...")
        html_msg = f"<p>Downloaded: <b>{os.path.basename(filename)}</b></p>"
        html_msg += "<p>Transcription & translations produced:</p><ul>"

        for key, desc in [("plain", "Plain transcript"),
                          ("timestamped", "Timestamped transcript"),
                          ("srt", "SRT subtitles"),
                          ("en", "English translation"),
                          ("en_ts", "English timestamped"),
                          ("en_srt", "English SRT"),
                          ("hi", "Hindi translation")]:
            if key in results and results.get(key):
                html_msg += f"<li>{desc}: <b>{os.path.basename(results[key])}</b></li>"

        html_msg += "</ul>"
        html_msg += f"<p>Processing time: <b>{results.get('seconds', 0):.1f} seconds</b></p>"

        progress(1.0, desc="Complete!")
        yield gr.update(value=results.get("plain"), visible=True), html_msg

    except Exception as e:
        print("General error in download_and_transcribe_stream:", traceback.format_exc())
        yield gr.update(visible=False, value=None), f"<i>Error during download/transcribe: {e}</i>"

def on_video_click(formats_json):
    return populate_quality_dropdown('video', formats_json)
def on_audio_click(formats_json):
    return populate_quality_dropdown('audio', formats_json)
def on_text_click(formats_json):
    return populate_quality_for_text(formats_json)

# ========== Gradio UI ==========

with gr.Blocks() as demo:
    gr.HTML("""
    <style>
      .orange-btn { background-color: orange !important; color: white !important; }
      h1.yt-title { color: orange; text-align: center; margin-top: 40px; }
      #thumb-img img { width: 500px !important; height: 300px !important; object-fit: contain !important; background:#f7f7f7; border-radius:6px; display:block; margin:0 auto; }
      .gr-row { gap:10px; }
      .small-note { font-size:0.9em; color:#666; text-align:center; margin-top:6px; }
    </style>
    """)

    gr.Markdown(
        "<h1 class='yt-title' style='color: orange; "
        "font-size: 64px; "
        "font-weight: 700; "
        "text-align: center; "
        "padding: 40px 0;'>YT-Dash</h1>"
    )

    url_input = gr.Textbox(label="", placeholder="Paste YouTube URL here...", lines=1)

    with gr.Row():
        submit_btn = gr.Button("Submit", elem_classes="orange-btn")
        clear_btn = gr.Button("Clear", elem_classes="orange-btn")

    with gr.Row():
        with gr.Column(scale=1):
            thumbnail = gr.Image(elem_id="thumb-img")
        with gr.Column(scale=2):
            video_title = gr.Markdown()
            with gr.Row():
                video_btn = gr.Button("Video", elem_classes="orange-btn", visible=False)
                audio_btn = gr.Button("Audio", elem_classes="orange-btn", visible=False)
                text_btn = gr.Button("Text", elem_classes="orange-btn", visible=False)
            quality_dropdown = gr.Dropdown(label="Choose quality (shows size)", choices=[], visible=False)
            with gr.Row():
                download_btn = gr.Button("Download / Show", elem_classes="orange-btn", visible=False)
                transcribe_btn = gr.Button("Download + Transcribe (Text)", elem_classes="orange-btn", visible=False)
            download_link_html = gr.HTML()
            file_output = gr.File(label="Downloaded / Transcript file", visible=False)

    hidden_formats = gr.Textbox(value="", visible=False)
    hidden_url = gr.Textbox(value="", visible=False)

    submit_btn.click(fn=submit_action,
                     inputs=url_input,
                     outputs=[thumbnail, video_title, hidden_formats, hidden_url, video_btn, audio_btn, text_btn, quality_dropdown, download_link_html, file_output])

    video_btn.click(fn=on_video_click, inputs=[hidden_formats], outputs=[quality_dropdown, download_btn, download_link_html])
    audio_btn.click(fn=on_audio_click, inputs=[hidden_formats], outputs=[quality_dropdown, download_btn, download_link_html])
    text_btn.click(fn=on_text_click, inputs=[hidden_formats], outputs=[quality_dropdown, transcribe_btn, download_link_html])

    download_btn.click(fn=download_selected_stream,
                       inputs=[quality_dropdown, hidden_formats, hidden_url],
                       outputs=[file_output, download_link_html])

    transcribe_btn.click(fn=download_and_transcribe_stream,
                         inputs=[quality_dropdown, hidden_formats, hidden_url],
                         outputs=[file_output, download_link_html])

    clear_btn.click(fn=clear_action, inputs=[], outputs=[thumbnail, video_title, hidden_formats, hidden_url, video_btn, audio_btn, text_btn, quality_dropdown, download_link_html, file_output])

    demo.queue(max_size=16)

demo.launch(share=False, show_error=True)
