# claimstream — Real-Time Audio Fact Checker

Streams microphone audio over WebSocket, transcribes it locally with Whisper, and verifies extracted claims against a FAISS-indexed Wikipedia corpus using DeBERTa-v3 NLI. All inference runs on-device — zero external API calls, zero network round-trips per claim.

## Problem

Speakers make claims in real time. Most fact-checking happens after the fact, manually, and doesn't scale. claimstream does it live:

1. Capture live audio from the browser microphone.
2. Transcribe to text locally with Whisper on CPU.
3. Extract and filter sentence-level claims from the rolling transcript.
4. Retrieve the top-5 most relevant Wikipedia passages via FAISS semantic search.
5. Run NLI (textual entailment) against each passage with DeBERTa-v3 and score the claim as **supported**, **refuted**, or **unverifiable**.

Two-side debate mode tracks accuracy per speaker and declares a winner on verifiable-claim hit rate.

## Architecture

```
                         browser mic
                              |
                 16 kHz PCM chunks (WebSocket)
                              |
                              v
               +-------------------------------+
               |  faster-whisper (base, int8)  |  Stage 1: STT
               +-------------------------------+
                              |
                              v
               +-------------------------------+
               |  rolling transcript buffer    |  Stage 2: sanitize
               |  (filler strip, punctuation)  |          + split
               +-------------------------------+
                              |
                              v
               +-------------------------------+
               |  sentence extractor +         |  Stage 3: claim
               |  non-claim filter             |          extraction
               +-------------------------------+
                              |
                              v
               +-------------------------------+
               |  all-MiniLM-L6-v2 encoder     |  Stage 4: retrieval
               |  -> FAISS (11,101 passages)   |          top-5
               +-------------------------------+
                              |
                              v
               +-------------------------------+
               |  DeBERTa-v3 mnli-fever-anli   |  Stage 5: NLI
               |  entail / neutral / contradict|
               +-------------------------------+
                              |
                              v
               +-------------------------------+
               |  per-side scoring engine      |  verdict + evidence
               |  (supported/refuted/unverif.) |  streamed to UI
               +-------------------------------+
```

Every stage runs locally. The FAISS index, passages file, and all three models sit on disk next to the process.

## Models

| Stage | Model | HuggingFace ID | Size |
|-------|-------|----------------|------|
| STT | Whisper (base, int8 CPU via faster-whisper) | [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) | ~140 MB |
| Retrieval | Sentence-Transformer | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | ~80 MB |
| NLI | DeBERTa-v3 (MNLI + FEVER + ANLI) | [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) | ~350 MB |

No fine-tuning. No API keys. Swap any model by changing one constant in `mymodel.py`.

## Knowledge Base

Wikipedia passages crawled via `build_index.py` from 108 seed topics (geography, science, history, technology, notable people, general knowledge). Articles are chunked into ~200-word passages and embedded with MiniLM; the resulting FAISS index is queried with cosine similarity at runtime.

- **Passages:** 11,101
- **Seed topics:** 108
- **Index:** `data/wiki_index.faiss` (~17 MB)
- **Corpus:** `data/wiki_passages.json` (~15 MB)

The NLI model was trained on MNLI + FEVER + ANLI — FEVER contributes Wikipedia-grounded fact-verification priors, which is why this pairing works well on this corpus.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# One-time: crawl Wikipedia and build the FAISS index (~3-5 min)
python build_index.py

# Run
python main.py
```

Open http://localhost:5000.

A `justfile` is included: `just setup`, `just build`, `just run`.

## File Structure

```
claimstream/
├── main.py              # Flask + Socket.IO entry, pipeline orchestration
├── mymodel.py           # Transcriber, Retriever, Verifier, FactChecker
├── preprocess.py        # Filler stripping, sanitization
├── debate.py            # Per-side scoring, winner logic
├── build_index.py       # Wikipedia crawler + FAISS index builder
├── requirements.txt
├── justfile
├── data/
│   ├── wiki_passages.json   # 11,101 passages
│   └── wiki_index.faiss     # MiniLM embeddings
├── templates/
│   └── index.html
└── static/
    └── diagram.html
```

## How It Works

1. **Audio capture** — browser streams 16 kHz PCM chunks over WebSocket (`audio_chunk` event, gevent async).
2. **Transcription** — faster-whisper `base` int8 with VAD filter (500 ms silence, 300 ms pad). Emits `new_text` back to the UI immediately.
3. **Buffering** — sanitized transcript accumulates per side. Verification triggers when at least 40 new chars arrive and the last verify was at least 12 s ago.
4. **Claim extraction** — split on sentence punctuation (fallback on conjunctions), strip non-claims (`I think`, `let me`, greetings), dedupe against already-verified claims.
5. **Retrieval** — MiniLM encodes the claim; FAISS returns top-5 passages by cosine similarity.
6. **NLI verification** — for each `(passage, claim)` pair, DeBERTa-v3 produces `[entail, neutral, contradict]` probabilities. Decision:
   - contradiction > 0.4 and > entailment → **refuted**
   - entailment > 0.3 → **supported**
   - otherwise → **unverifiable**
7. **Scoring** — `debate.py` tracks supported / refuted / unverifiable per side; winner is the side with higher accuracy on verifiable claims.

## Design Notes

- **Why contradiction wins.** NLI returns "neutral" for most true claims because retrieved evidence rarely paraphrases the claim word-for-word. Treating neutral-with-relevant-evidence as implicit support, and demanding a strong contradiction signal to refute, keeps false-positive refutations low.
- **Why CPU-only Whisper.** `faster-whisper` with `int8` quantization makes the `base` model real-time on a laptop CPU. No GPU dependency means the whole stack runs in a 2 GB container.
- **Why FAISS over a vector DB.** 11,101 passages fits in memory. FAISS `read_index` + one `search()` call is roughly 1 ms per query on CPU. A hosted vector DB would add latency and a network dependency for no benefit at this scale.
