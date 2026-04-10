"""
main.py — ConvoAI: Real-Time Space Debate Fact Checker

A real-time audio debate fact-checking system that:
1. Captures live microphone audio from a browser
2. Transcribes speech locally using Whisper (faster-whisper)
3. Verifies claims against a 9,700+ fact dataset (FEVER/Wikipedia)
4. Tracks scores per debate side and shows corrections

All models run locally — no API calls.

Usage:
    python main.py

Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import re
from mymodel import Transcriber, FactChecker
from preprocess import extract_claims, is_filler
from debate import Debate

# ============================================================
# Flask App Setup
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "convoai-debate"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# ============================================================
# Load Models (local, no API)
# ============================================================

print("=" * 60)
print("ConvoAI — Space Debate Fact Checker")
print("=" * 60)
print("\nLoading models (this may take a moment on first run)...\n")

print("[1/2] Speech-to-Text (Whisper)")
transcriber = Transcriber(model_size="tiny", device="cpu")

print("\n[2/2] Fact Verification (Sentence-Transformers + FEVER)")
checker = FactChecker()

print("\n" + "=" * 60)
print("All models loaded. Open http://localhost:5000")
print("=" * 60 + "\n")

# ============================================================
# Debate Session + Rolling Transcript Buffer
# ============================================================

debate = Debate()

# Per-side rolling buffer of partial transcripts
# Accumulates text as chunks come in. When we detect a sentence boundary,
# we fact-check the completed sentence and keep the remainder in the buffer.
buffer = {"a": "", "b": ""}

# Track recently processed sentences to avoid duplicates
processed_sentences = {"a": set(), "b": set()}

SENTENCE_END = re.compile(r'[.!?]+\s*')


def flush_sentences(side: str, force: bool = False) -> list[str]:
    """Pull complete sentences out of the buffer for the given side.

    Returns a list of new sentences to fact-check.
    If force=True, flushes remaining buffer even without sentence-ending punctuation.
    """
    text = buffer[side].strip()
    if not text:
        return []

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    if force:
        # Flush everything
        for p in parts:
            p = p.strip()
            if p and len(p) >= 10:
                sentences.append(p)
        buffer[side] = ""
    else:
        # Only take sentences that end with punctuation; keep the last incomplete part
        if text.endswith(('.', '!', '?')):
            # Last sentence is complete
            for p in parts:
                p = p.strip()
                if p and len(p) >= 10:
                    sentences.append(p)
            buffer[side] = ""
        else:
            # Last part is incomplete — process all but the last
            for p in parts[:-1]:
                p = p.strip()
                if p and len(p) >= 10:
                    sentences.append(p)
            buffer[side] = parts[-1] if parts else ""

    # Dedupe (in case Whisper re-transcribed the same audio)
    unique = []
    for s in sentences:
        key = s.lower().strip()
        if key not in processed_sentences[side] and not is_filler(s):
            processed_sentences[side].add(key)
            unique.append(s)

    return unique


def reset_buffer():
    buffer["a"] = ""
    buffer["b"] = ""
    processed_sentences["a"].clear()
    processed_sentences["b"].clear()

# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# WebSocket Events
# ============================================================

@socketio.on("connect")
def on_connect():
    emit("status", {"message": "Connected. Models ready."})


@socketio.on("set_sides")
def on_set_sides(data):
    global debate
    debate = Debate(
        side_a_name=data.get("side_a", "Side A"),
        side_b_name=data.get("side_b", "Side B"),
    )
    reset_buffer()
    emit("debate_reset", debate.get_full_summary())


@socketio.on("set_active_side")
def on_set_active_side(data):
    side = data.get("side", "a")
    debate.set_active_side(side)
    emit("side_changed", {"side": side, "name": debate.sides[side].name})


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    """Process incoming audio chunk: transcribe -> buffer -> detect sentences -> fact-check.

    This is called continuously every ~3 seconds while recording. Text accumulates
    in a rolling buffer; complete sentences are extracted and fact-checked as they form.
    """
    audio_bytes = data.get("audio")
    side = data.get("side", debate.active_side or "a")

    if not audio_bytes:
        return

    # Transcribe the chunk (Whisper tiny — fast)
    text = transcriber.transcribe_audio(audio_bytes)
    if not text:
        return

    # Append to the rolling buffer for this side
    buffer[side] = (buffer[side] + " " + text).strip()

    # Show live partial transcript immediately
    emit("transcription", {"text": buffer[side], "side": side, "partial": True})

    # Extract any complete sentences from the buffer
    sentences = flush_sentences(side)

    for claim in sentences:
        result = checker.check(claim)
        debate.add_claim(claim, result, side=side)

        emit("fact_result", {
            "claim": claim,
            "result": result,
            "side": side,
            "scores": {
                k: s.summary() for k, s in debate.sides.items()
            },
        })


@socketio.on("end_side")
def on_end_side(data):
    side = data.get("side", debate.active_side or "a")

    # Force-flush any remaining buffer text
    remaining = flush_sentences(side, force=True)
    for claim in remaining:
        result = checker.check(claim)
        debate.add_claim(claim, result, side=side)
        emit("fact_result", {
            "claim": claim,
            "result": result,
            "side": side,
            "scores": {k: s.summary() for k, s in debate.sides.items()},
        })

    claims = debate.get_side_claims(side)
    summary = debate.get_side_summary(side)
    emit("side_summary", {
        "side": side,
        "summary": summary,
        "claims": claims,
    })


@socketio.on("end_debate")
def on_end_debate():
    full = debate.get_full_summary()
    full["all_claims"] = {
        k: debate.get_side_claims(k) for k in debate.sides
    }
    emit("debate_summary", full)


@socketio.on("reset")
def on_reset():
    debate.reset()
    reset_buffer()
    emit("debate_reset", debate.get_full_summary())


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
