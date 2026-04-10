"""
mymodel.py — Model loading and inference for ConvoAI Debate Fact Checker.

Three pre-trained HuggingFace models, all running locally:
1. Whisper (faster-whisper) — speech-to-text
2. all-MiniLM-L6-v2 — semantic retrieval (find relevant facts)
3. DeBERTa-v3-base-mnli-fever-anli — NLI fact verification (trained on FEVER)

No custom training. All models work out of the box.
"""

import json
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Speech-to-Text (Whisper)
# ============================================================

class Transcriber:
    """Local STT using faster-whisper. Tiny model for real-time CPU inference."""

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        print(f"  Loading Whisper '{model_size}' (device={device}, {compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"  Whisper ready.")

    def transcribe_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio, beam_size=5, language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ============================================================
# Fact Verification (Retrieval + NLI)
# ============================================================

# Pre-trained models from HuggingFace — no fine-tuning needed
RETRIEVAL_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class FactChecker:
    """Two-stage fact checker using pre-trained HuggingFace models.

    Stage 1 — RETRIEVAL: Sentence-transformer finds the top-3 most similar
    facts from the database via cosine similarity.

    Stage 2 — VERIFICATION: DeBERTa NLI cross-encoder classifies whether the
    speaker's claim ENTAILS or CONTRADICTS the matched fact. This model is
    trained on MNLI + FEVER + ANLI — specifically designed for fact verification.

    Why two stages? Cosine similarity can't detect negation.
    "Jupiter is NOT the largest" scores ~90% similar to "Jupiter IS the largest".
    The NLI model catches the contradiction.
    """

    def __init__(self, dataset_path: str = "data/space_facts.json", threshold: float = 0.45):
        self.threshold = threshold

        # Stage 1: Retrieval model
        print(f"  Loading retrieval model: {RETRIEVAL_MODEL}")
        self.retriever = SentenceTransformer(RETRIEVAL_MODEL)

        # Fact database
        print(f"  Loading fact dataset: {dataset_path}")
        self.facts = self._load_dataset(dataset_path)
        print(f"  {len(self.facts)} facts loaded")

        # Pre-encode all facts for fast similarity search
        print(f"  Encoding fact embeddings...")
        claims = [f["claim"] for f in self.facts]
        self.embeddings = self.retriever.encode(claims, normalize_embeddings=True, show_progress_bar=True)

        # Stage 2: NLI model (trained on MNLI + FEVER + ANLI)
        print(f"  Loading NLI model: {NLI_MODEL}")
        self.nli = CrossEncoder(NLI_MODEL)
        print(f"  Fact checker ready. (threshold={self.threshold})")

    def _load_dataset(self, path: str) -> list[dict]:
        with open(Path(path)) as f:
            return json.load(f)["facts"]

    def _get_nli_label(self, scores) -> tuple[int, float]:
        """Get NLI label and confidence from model output.

        cross-encoder/nli-deberta-v3-small outputs: [contradiction, entailment, neutral]
        """
        label = int(np.argmax(scores))
        conf = float(scores[label])
        return label, conf

    def check(self, claim: str) -> dict:
        """Fact-check a single claim.

        Returns: {status, confidence, message, matched_claim, source, category}
        """
        # Stage 1: Find top-3 similar facts
        emb = self.retriever.encode([claim], normalize_embeddings=True)
        sims = np.dot(self.embeddings, emb.T).flatten()
        top_k = min(3, len(self.facts))
        top_idx = np.argsort(sims)[-top_k:][::-1]
        top_sim = float(sims[top_idx[0]])

        if top_sim < self.threshold:
            return {
                "status": "unverifiable",
                "confidence": round(top_sim, 3),
                "message": "No matching fact found in the dataset.",
                "matched_claim": None,
                "category": None,
            }

        # Stage 2: NLI on each candidate — pick strongest non-neutral match
        best = None
        best_score = -1

        for idx in top_idx:
            sim = float(sims[idx])
            if sim < self.threshold:
                break

            fact = self.facts[idx]
            scores = self.nli.predict([(claim, fact["claim"])])[0]
            label, conf = self._get_nli_label(scores)

            # Labels: 0=contradiction, 1=entailment, 2=neutral
            if label == 2:
                continue  # skip neutral

            combined = sim * 0.4 + conf * 0.6
            if combined > best_score:
                best_score = combined
                best = {"fact": fact, "sim": sim, "label": label, "conf": conf}

        if best is None:
            return {
                "status": "unverifiable",
                "confidence": round(top_sim, 3),
                "message": f"Closest: \"{self.facts[top_idx[0]]['claim'][:80]}...\" but unclear relationship.",
                "matched_claim": self.facts[top_idx[0]]["claim"],
                "source": self.facts[top_idx[0]].get("source", ""),
                "category": self.facts[top_idx[0]].get("category", ""),
            }

        fact = best["fact"]
        is_entailment = (best["label"] == 1)
        fact_true = fact["verdict"]

        # Logic matrix:
        # entailment  + true fact  → SUPPORTED (speaker affirms a truth)
        # entailment  + false fact → REFUTED   (speaker affirms a misconception)
        # contradict  + true fact  → REFUTED   (speaker denies a truth)
        # contradict  + false fact → SUPPORTED (speaker denies a misconception)
        if (is_entailment and fact_true) or (not is_entailment and not fact_true):
            status = "supported"
            msg = f"Supported by: \"{fact['claim']}\""
        else:
            status = "refuted"
            if fact_true:
                msg = f"Contradicts known fact: \"{fact['claim']}\""
            else:
                msg = f"Misconception. {fact.get('correction', 'This claim is false.')}"

        return {
            "status": status,
            "confidence": round(best["sim"], 3),
            "message": msg,
            "matched_claim": fact["claim"],
            "source": fact.get("source", ""),
            "category": fact.get("category", ""),
        }

    def check_multiple(self, claims: list[str]) -> list[dict]:
        return [self.check(c) for c in claims]
