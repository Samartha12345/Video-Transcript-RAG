"""Load English captions or a UTF-8 transcript supplied by the user."""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def video_id(value: str) -> str:
    value = value.strip()
    pattern = r"[A-Za-z0-9_-]{11}"
    if re.fullmatch(pattern, value):
        return value
    parsed = urlparse(value if "://" in value else "https://" + value)
    host = parsed.hostname
    parts = parsed.path.strip("/").split("/")
    candidate = ""
    if parsed.scheme in {"http", "https"}:
        if host == "youtu.be" and len(parts) == 1:
            candidate = parts[0]
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if parts == ["watch"]:
                candidate = parse_qs(parsed.query).get("v", [""])[0]
            elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
                candidate = parts[1]
    if not re.fullmatch(pattern, candidate):
        raise ValueError("Enter a YouTube URL or an 11-character video ID.")
    return candidate


def load_transcript(video: str | None, file: Path | None) -> tuple[str, str]:
    if (video is None) == (file is None):
        raise ValueError("Provide exactly one video or transcript file.")
    if file is not None:
        text = file.read_text(encoding="utf-8-sig")
        source = file.name
    else:
        identifier = video_id(video)
        from youtube_transcript_api import YouTubeTranscriptApi

        captions = YouTubeTranscriptApi().fetch(identifier, languages=["en"])
        text = " ".join(snippet.text.strip() for snippet in captions)
        source = f"https://www.youtube.com/watch?v={identifier}"
    if not text.strip():
        raise ValueError("The transcript is empty.")
    return text.strip(), source
