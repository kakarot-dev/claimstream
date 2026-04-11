# ConvoAI — Real-Time Debate Fact Checker

# Setup everything from scratch
setup:
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    @echo "Setup done. Run: just build then just run"

# Build Wikipedia FAISS index (one-time)
build:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python build_index.py

# Run the app (default port 5000)
run port="5000":
    CUDA_VISIBLE_DEVICES="" .venv/bin/python main.py {{port}}

# Quick start (build + run)
start: build run

# Clean generated files
clean:
    rm -rf data/wiki_passages.json data/wiki_index.faiss __pycache__

# Rebuild index from scratch
rebuild: clean build

# Push to GitHub
push msg="update":
    git add -A && git commit -m "{{msg}}" && git push origin master
