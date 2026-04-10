# ConvoAI — Real-Time Debate Fact Checker

A real-time audio debate fact-checking system that listens to speakers, transcribes speech locally using Whisper, and verifies claims against 101,897 Wikipedia-verified facts from the FEVER dataset using semantic retrieval and Natural Language Inference.

## Problem Statement

In debates, speakers make claims that may be incorrect. This system provides real-time fact-checking by:
1. Capturing live audio from the browser microphone
2. Converting speech to text using a local Whisper model
3. Matching claims against 101,897 verified facts using semantic similarity + NLI
4. Scoring each debate side by factual accuracy

## Models Used

| Model | HuggingFace ID | Purpose | Size |
|-------|---------------|---------|------|
| Whisper (tiny) | openai/whisper-tiny | Speech-to-Text | ~75 MB |
| Sentence-Transformer | all-MiniLM-L6-v2 | Semantic retrieval | ~80 MB |
| DeBERTa NLI | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | Fact verification | ~350 MB |

All models run entirely locally — no API calls. No custom training needed.

## Dataset

- **Source**: FEVER (Fact Extraction and VERification) — full dataset from HuggingFace
- **Size**: 101,897 claims (73,784 true, 28,113 false)
- **Domain**: General knowledge (all Wikipedia topics)
- **Format**: JSON with claim, verdict, source fields

## Setup and Execution

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open http://localhost:5000 in your browser.

## File Structure

```
convoai/
├── main.py              # Main entry point
├── mymodel.py           # Model loading: Whisper + SentenceTransformer + DeBERTa NLI
├── preprocess.py        # Sentence splitting, filler filtering
├── debate.py            # Debate scoring logic
├── requirements.txt     # Dependencies
├── README.md            # This file
├── dataset_sample.csv   # 100-row sample
├── results_output.txt   # Example output
├── data/
│   └── facts.json       # Full FEVER dataset (101,897 facts)
└── templates/
    └── index.html       # Web UI
```

## How It Works

1. **Audio Capture**: Browser streams 2.5s PCM chunks via WebSocket
2. **Speech-to-Text**: Whisper tiny transcribes in real-time on CPU
3. **Rolling Buffer**: Text accumulates, sentences extracted on punctuation
4. **Retrieval**: Sentence-transformer finds top-3 similar facts (cosine similarity)
5. **NLI Verification**: DeBERTa (trained on FEVER) classifies entailment/contradiction
6. **Scoring**: Tracks supported/refuted per side, determines winner by accuracy
