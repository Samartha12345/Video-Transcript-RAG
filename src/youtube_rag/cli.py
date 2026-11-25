"""Command-line entry point for transcript question answering."""

import argparse
import sys
from pathlib import Path

from .transcripts import load_transcript


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="YouTube URL or video ID with English captions")
    source.add_argument("--transcript", type=Path, help="Local UTF-8 transcript file")
    parser.add_argument("--question", required=True, help="Question about the transcript")
    parser.add_argument("--show-passages", action="store_true", help="Print retrieved evidence")
    args = parser.parse_args(argv)
    question = args.question.strip()
    if not question:
        parser.error("Enter a non-empty question.")
    try:
        text, source_name = load_transcript(args.video, args.transcript)
    except Exception as exc:
        print(f"Could not load transcript: {exc}", file=sys.stderr)
        if args.video:
            print("You can also supply a local caption file with --transcript.", file=sys.stderr)
        return 1
    from .generation import answer_question
    from .retrieval import index_transcript

    passages = index_transcript(text, source_name).invoke(question)
    print(answer_question(question, passages))
    if args.show_passages:
        print(f"\nSource: {source_name}")
        for doc in passages:
            print(f"\n[Passage {doc.metadata['passage']}] {doc.page_content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
