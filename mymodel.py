"""
mymodel.py — Model loading and NLP pipeline for ConvoAI.

Stage 1: Whisper STT (faster-whisper)
Stage 4: Evidence retrieval (sentence-transformers + FAISS + Wikipedia)
Stage 5: NLI verification (DeBERTa-v3 trained on MNLI/FEVER/ANLI)
"""

import json
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import faiss


# ============================================================
# Stage 1: Speech-to-Text (Whisper)
# ============================================================

class Transcriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        print(f"  Loading Whisper '{model_size}' ({device}, {compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"  Whisper ready.")

    def transcribe_audio(self, audio_bytes, sample_rate=16000):
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < 1600:
            return ""
        segments, _ = self.model.transcribe(
            audio, beam_size=5, language="en", vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ============================================================
# Stage 4: Evidence Retrieval (FAISS + Sentence-Transformers)
# ============================================================

RETRIEVAL_MODEL = "all-MiniLM-L6-v2"
PASSAGES_PATH = "data/wiki_passages.json"
INDEX_PATH = "data/wiki_index.faiss"
TOP_K = 5


class Retriever:
    """Retrieve relevant Wikipedia passages for a claim using FAISS."""

    def __init__(self):
        print(f"  Loading retrieval model: {RETRIEVAL_MODEL}")
        self.model = SentenceTransformer(RETRIEVAL_MODEL, device="cpu")

        print(f"  Loading Wikipedia passages...")
        with open(PASSAGES_PATH) as f:
            self.passages = json.load(f)
        print(f"  {len(self.passages)} passages loaded")

        print(f"  Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)
        print(f"  FAISS index ready ({self.index.ntotal} vectors)")

    def search(self, claim, top_k=TOP_K):
        """Find top-k Wikipedia passages most relevant to the claim."""
        emb = self.model.encode([claim], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.passages):
                p = self.passages[idx]
                results.append({
                    "text": p["text"],
                    "title": p.get("title", ""),
                    "source": p.get("source", "Wikipedia"),
                    "score": float(score),
                })
        return results


# ============================================================
# Stage 5: NLI Verification (DeBERTa)
# ============================================================

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class Verifier:
    """Verify a claim against evidence using NLI (textual entailment)."""

    def __init__(self):
        print(f"  Loading NLI model: {NLI_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()
        # Labels: {0: entailment, 1: neutral, 2: contradiction}
        print(f"  NLI model ready.")

    def verify(self, claim, evidence_passages):
        """Check claim against evidence passages.

        Returns: {"verdict": "supported"|"refuted"|"unverifiable",
                  "confidence": float, "evidence": str, "source": str, "reason": str}
        """
        best_entail = {"score": 0, "passage": None}
        best_contra = {"score": 0, "passage": None}

        for passage in evidence_passages:
            inputs = self.tokenizer(
                claim, passage["text"],
                return_tensors="pt", truncation=True, max_length=512,
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()

            entail_score = float(probs[0])
            contra_score = float(probs[2])

            if entail_score > best_entail["score"]:
                best_entail = {"score": entail_score, "passage": passage}
            if contra_score > best_contra["score"]:
                best_contra = {"score": contra_score, "passage": passage}

        # Decide verdict based on strongest signal
        if best_contra["score"] > 0.7:
            return {
                "verdict": "refuted",
                "confidence": round(best_contra["score"], 3),
                "evidence": best_contra["passage"]["text"][:200],
                "source": best_contra["passage"].get("source", ""),
                "reason": f"Contradicted by Wikipedia ({best_contra['passage'].get('title', '')})",
            }
        elif best_entail["score"] > 0.7:
            return {
                "verdict": "supported",
                "confidence": round(best_entail["score"], 3),
                "evidence": best_entail["passage"]["text"][:200],
                "source": best_entail["passage"].get("source", ""),
                "reason": f"Supported by Wikipedia ({best_entail['passage'].get('title', '')})",
            }
        elif best_entail["score"] > 0.5:
            return {
                "verdict": "supported",
                "confidence": round(best_entail["score"], 3),
                "evidence": best_entail["passage"]["text"][:200],
                "source": best_entail["passage"].get("source", ""),
                "reason": f"Likely supported ({best_entail['passage'].get('title', '')})",
            }
        else:
            return {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "evidence": "",
                "source": "",
                "reason": "No strong evidence found in Wikipedia",
            }


# ============================================================
# Full Pipeline
# ============================================================

class FactChecker:
    """Combines retrieval + NLI for end-to-end fact checking."""

    def __init__(self):
        self.retriever = Retriever()
        self.verifier = Verifier()

    def check(self, claim):
        """Check a single claim: retrieve evidence → NLI verify."""
        # Stage 4: Retrieve evidence
        evidence = self.retriever.search(claim)

        if not evidence:
            return {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "evidence": "",
                "source": "",
                "reason": "No relevant evidence found",
            }

        # Stage 5: NLI verification
        result = self.verifier.verify(claim, evidence)
        return result
