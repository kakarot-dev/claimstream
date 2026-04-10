# ConvoAI — Real-Time Space Debate Fact Checker

A real-time audio debate fact-checking system that listens to speakers, transcribes their speech, and verifies claims against a dataset of 9,700+ space/astronomy facts sourced from the FEVER (Fact Extraction and VERification) dataset.

## Problem Statement

In debates and discussions about space and astronomy, speakers often make claims that may be factually incorrect. This project provides real-time fact-checking by:
1. Capturing live audio from the browser microphone
2. Converting speech to text using a local Whisper model
3. Matching claims against a verified fact database using semantic similarity
4. Scoring each debate side by factual accuracy

## Architecture

```
Browser Microphone
       |
       v
[Audio Capture (WebSocket)] ──> [Whisper STT Model] ──> Raw Text
                                                            |
                                                            v
                                                   [Preprocessing]
                                                   (sentence split,
                                                    filler removal)
                                                            |
                                                            v
                                                   [Sentence-Transformer]
                                                   (encode claim →
                                                    cosine similarity
                                                    vs 9,700 facts)
                                                            |
                                                            v
                                                   [Verdict: Supported /
                                                    Refuted / Unverifiable]
                                                            |
                                                            v
                                                   [Debate Tracker]
                                                   (scores per side)
                                                            |
                                                            v
                                                   [Web UI Dashboard]
```

## Models Used

| Model | Source | Purpose | Size |
|-------|--------|---------|------|
| `faster-whisper` (small) | HuggingFace / OpenAI | Speech-to-Text | ~461 MB |
| `all-MiniLM-L6-v2` | HuggingFace / sentence-transformers | Sentence embeddings for fact matching | ~80 MB |
| Fine-tuned variant | Trained locally via `train.py` | Domain-specific embeddings for space claims | ~80 MB |

All models are downloaded from HuggingFace and run **entirely locally** — no API calls.

## Dataset

- **Source**: FEVER (Fact Extraction and VERification) dataset — Wikipedia-verified claims
- **Size**: 9,747 facts (7,418 true, 2,329 false)
- **Domain**: Space, astronomy, cosmology, planetary science, space exploration
- **Format**: JSON with claim, verdict (true/false), source, and category fields
- **Sample**: See `dataset_sample.csv`

## Setup and Execution

### Prerequisites
- Python 3.10+
- Microphone access (for live demo)
- ~1 GB disk space for models

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

Then open **http://localhost:5000** in your browser.

### Optional: Fine-tune the model

```bash
python train.py
```

This fine-tunes the sentence-transformer on the space dataset (~20 min on CPU).

## File Structure

```
convoai/
├── main.py                 # Main entry point (python main.py)
├── mymodel.py              # Model loading: Whisper + Sentence-Transformer
├── preprocess.py           # Data preprocessing: sentence splitting, filler filtering
├── debate.py               # Debate state tracking and scoring
├── train.py                # Fine-tuning script for sentence-transformer
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── dataset_sample.csv      # Sample of the fact dataset
├── results_output.txt      # Example output from the system
├── data/
│   └── space_facts.json    # Full fact dataset (9,747 entries)
├── models/
│   └── space-fact-checker/ # Fine-tuned model (after running train.py)
└── templates/
    └── index.html          # Web UI
```

## How It Works

1. **Audio Capture**: Browser captures microphone audio via MediaRecorder API, streams PCM chunks to the server via WebSocket.

2. **Speech-to-Text**: `faster-whisper` (Whisper small model, int8 quantized) transcribes audio chunks locally. VAD (Voice Activity Detection) filters silence.

3. **Preprocessing**: Transcribed text is split into sentences. Filler speech ("um", "hello", "test") is filtered out.

4. **Fact Checking**: Each claim is encoded using `sentence-transformers` (`all-MiniLM-L6-v2` or fine-tuned variant). Cosine similarity is computed against all 9,747 pre-encoded facts. If similarity > 0.45:
   - Matched fact is TRUE → claim is **Supported**
   - Matched fact is FALSE → claim is **Refuted** (with correction shown)
   - Below threshold → **Unverifiable**

5. **Scoring**: Each debate side tracks supported/refuted/unverifiable counts. Accuracy = supported / (supported + refuted). Winner is determined by accuracy.

## Evaluation

The system's effectiveness depends on:
- **Whisper accuracy**: The 'small' model provides good transcription for clear speech
- **Semantic matching**: The sentence-transformer captures meaning, not just keywords
- **Dataset coverage**: 9,747 facts from FEVER/Wikipedia cover broad space topics
- **Fine-tuning**: Training on domain-specific paraphrases improves matching accuracy
