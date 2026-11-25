import tempfile
import unittest
from pathlib import Path

from youtube_rag.transcripts import load_transcript, video_id


class TranscriptTests(unittest.TestCase):
    def test_supported_video_inputs(self):
        value = "Gfr50f6ZBvo"
        for url in (value, f"https://www.youtube.com/watch?v={value}&t=30",
                    f"youtu.be/{value}", f"https://m.youtube.com/shorts/{value}"):
            with self.subTest(url=url):
                self.assertEqual(video_id(url), value)

    def test_reject_unrelated_links_and_missing_ids(self):
        for url in ("https://example.com/watch?v=Gfr50f6ZBvo", "youtube.com/watch", "abc"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                video_id(url)

    def test_utf8_file_and_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.txt"
            path.write_text("Speaker’s notes", encoding="utf-8-sig")
            self.assertEqual(load_transcript(None, path), ("Speaker’s notes", "captions.txt"))
            path.write_text("  ", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_transcript(None, path)
