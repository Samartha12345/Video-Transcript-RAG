# YouTube Transcript RAG

Ask a question about a YouTube video and get an answer based on its English captions. This Python project turns the original RAG notebook into a reusable package, a command-line tool and a short Jupyter walkthrough. It also accepts plain-text transcripts.

## Quick start

Install Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Clone the repository and open its folder. These commands work in PowerShell and Bash.

```sh
git clone https://github.com/Samartha12345/Video-Transcript-RAG.git
cd Video-Transcript-RAG
uv sync --locked
uv run youtube-rag --transcript examples/sample_transcript.txt --question "How often does the vibration sensor collect readings?" --show-passages
```

The sample transcript says **once per second**. The first run downloads the embedding model and language model from Hugging Face. Later runs reuse the model cache. The models run locally on CPU. You do not need an API key. On a laptop, generating the first answer can take some time.

To ask about the video used in the original notebook, run this command (the URL or the 11-character video ID both work).

```sh
uv run youtube-rag --video "https://www.youtube.com/watch?v=Gfr50f6ZBvo" --question "What does the speaker say about neural networks?" --show-passages
```

YouTube must make English captions available to your connection. If captions are disabled or a video request is blocked, use `--transcript` with a UTF-8 text file. Omit `--show-passages` to print just the answer.

## How the answer is built

| Stage | What happens |
| --- | --- |
| Load | Fetch English caption snippets with the YouTube Transcript API, or read a local transcript. |
| Split | LangChain splits the transcript into passages of up to 1,000 characters, targeting a 200-character overlap so nearby sentences retain context. |
| Index | Sentence Transformers encodes each passage with `all-MiniLM-L6-v2`. An in-memory ChromaDB collection stores its embedding, source and passage number. |
| Retrieve | Embed the question and retrieve up to four passages with similar meaning. |
| Generate | Send the passages and question to `Qwen/Qwen2.5-0.5B-Instruct` through Hugging Face Transformers on CPU. The prompt asks the model to acknowledge missing information. |

Use the printed passages to check an answer. Retrieval always returns the closest available passages, even for an unrelated question. The model can still give an unsupported answer. The tool works with spoken captions, so details shown only on screen are outside its scope.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/youtube_rag/transcripts.py` | Parse YouTube links and IDs, fetch captions and read transcript files. |
| `src/youtube_rag/retrieval.py` | Split text, embed passages and build an isolated Chroma collection. |
| `src/youtube_rag/generation.py` | Format the evidence and run the local Qwen model. |
| `src/youtube_rag/cli.py` | Connect loading, retrieval and generation to the command line. |
| `notebooks/RAG_walkthrough.ipynb` | Run the pipeline step by step and inspect the evidence. |
| `examples/sample_transcript.txt` | Try the project without depending on YouTube captions. |
| `tests/` | Run repeatable checks without downloading model weights. |
| `pyproject.toml`, `uv.lock` | Package metadata, CLI entry point and resolved dependencies. |

## Work in VS Code or Jupyter

```sh
uv sync --locked --extra notebook
```

Open `notebooks/RAG_walkthrough.ipynb` in VS Code, select the project's `.venv` Python kernel and run the cells in order. Change the question in the last two cells to reuse the same index. A new CLI invocation builds a fresh index.

## Run tests

```sh
uv run python -m unittest discover -s tests -v
```
