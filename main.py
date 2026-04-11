"""
main.py — ConvoAI: Real-Time Debate Fact Checker

5-stage NLP pipeline:
  1. Whisper STT (audio → text)
  2. Sanitization (clean filler, fix formatting)
  3. Claim extraction (split sentences, filter opinions)
  4. Evidence retrieval (FAISS + Wikipedia)
  5. NLI verification (DeBERTa entailment/contradiction)

Usage: python main.py
"""

import time
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from mymodel import Transcriber, FactChecker
from preprocess import sanitize, is_filler
from debate import Debate

app = Flask(__name__)
app.config["SECRET_KEY"] = "convoai"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

print("=" * 50)
print("ConvoAI — Debate Fact Checker")
print("=" * 50)

print("\n[1/3] Whisper STT")
transcriber = Transcriber(model_size="small", device="cpu")

print("\n[2/3] Wikipedia Evidence Retrieval")
print("[3/3] DeBERTa NLI Verification")
checker = FactChecker()

print("\n" + "=" * 50)
print("Ready. http://localhost:5000")
print("=" * 50 + "\n")

debate = Debate()
transcripts = {"a": "", "b": ""}
verified_claims = {"a": [], "b": []}
last_verify = {"a": 0, "b": 0}
last_verify_len = {"a": 0, "b": 0}

VERIFY_INTERVAL = 12


def reset_state():
    for s in ("a", "b"):
        transcripts[s] = ""
        verified_claims[s] = []
        last_verify[s] = 0
        last_verify_len[s] = 0


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    emit("status", {"msg": "ready"})


@socketio.on("set_sides")
def on_set_sides(data):
    global debate
    debate = Debate(data.get("side_a", "Side A"), data.get("side_b", "Side B"))
    reset_state()
    emit("debate_reset", {})


@socketio.on("set_active_side")
def on_set_active_side(data):
    debate.set_active_side(data.get("side", "a"))


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    audio_raw = data.get("audio")
    side = data.get("side", debate.active_side or "a")
    if not isinstance(audio_raw, (bytes, bytearray)):
        return

    audio_bytes = bytes(audio_raw)

    # Resample if needed
    client_rate = data.get("sampleRate", 16000)
    if client_rate != 16000:
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        ratio = 16000 / client_rate
        indices = np.arange(0, len(samples), 1 / ratio).astype(int)
        indices = indices[indices < len(samples)]
        audio_bytes = samples[indices].tobytes()

    # Stage 1: Transcribe
    raw = transcriber.transcribe_audio(audio_bytes)
    if not raw:
        return

    # Stage 2: Sanitize
    clean = sanitize(raw)
    if not clean or is_filler(clean):
        return

    # Accumulate
    transcripts[side] = (transcripts[side] + " " + clean).strip()
    emit("new_text", {"side": side, "text": clean})

    # Auto-verify periodically
    now = time.time()
    new_chars = len(transcripts[side]) - last_verify_len[side]
    if new_chars > 40 and now - last_verify[side] > VERIFY_INTERVAL:
        run_verify(side)


@socketio.on("verify_now")
def on_verify_now(data):
    run_verify(data.get("side", debate.active_side or "a"))


@socketio.on("end_side")
def on_end_side(data):
    side = data.get("side", debate.active_side or "a")
    emit("loading", {"msg": "Verifying claims..."})

    # Final verify
    if transcripts[side] and len(transcripts[side]) > last_verify_len[side]:
        run_verify(side)

    summary = debate.get_side_summary(side)
    emit("turn_report", {
        "side": side,
        "summary": summary,
        "claims": verified_claims[side],
    })


@socketio.on("end_debate")
def on_end_debate():
    full = debate.get_full_summary()
    emit("debate_summary", full)


@socketio.on("reset")
def on_reset():
    debate.reset()
    reset_state()
    emit("debate_reset", {})


def run_verify(side):
    """Run stages 3-5 on new transcript text."""
    new_text = transcripts[side][last_verify_len[side]:]
    if not new_text.strip() or len(new_text.split()) < 4:
        return

    last_verify[side] = time.time()
    last_verify_len[side] = len(transcripts[side])

    # Stage 3: Extract claims (split into sentences)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', new_text)
    # Also treat the whole text as a claim if no punctuation
    if len(sentences) == 1 and not any(new_text.rstrip().endswith(p) for p in '.!?'):
        sentences = [new_text]

    claims = [s.strip() for s in sentences if len(s.strip()) >= 15 and not is_filler(s)]
    print(f"[VERIFY] side={side}, {len(claims)} claims from {len(new_text)} chars")

    for claim in claims:
        # Stages 4+5: Retrieve + Verify
        result = checker.check(claim)
        print(f"  [{result['verdict']}] {claim[:50]} — {result['reason'][:40]}")

        verified_claims[side].append({
            "text": claim,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "evidence": result.get("evidence", ""),
            "source": result.get("source", ""),
        })

        if result["verdict"] in ("supported", "refuted"):
            debate.add_claim(claim, {
                "status": "supported" if result["verdict"] == "supported" else "refuted",
                "message": result["reason"],
            }, side=side)

        emit("highlight", {
            "side": side,
            "claim": {
                "text": claim,
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "reason": result["reason"],
            },
            "scores": {k: s.summary() for k, s in debate.sides.items()},
        })


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
