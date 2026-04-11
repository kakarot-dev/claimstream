"""
build_index.py — Build FAISS index from Wikipedia articles.

Crawls Wikipedia articles on common debate topics, splits into passages,
encodes with sentence-transformer, and saves FAISS index + passages.

Run once: python build_index.py
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import wikipediaapi

TOPICS = [
    # Geography & Countries
    "Rome", "Italy", "Germany", "France", "United Kingdom", "United States",
    "China", "India", "Japan", "Brazil", "Russia", "Australia", "Spain",
    "Capital city", "Europe", "Asia", "Africa",

    # Science
    "Earth", "Solar System", "Planet", "Moon", "Sun", "Mars", "Jupiter",
    "Saturn", "Venus", "Mercury (planet)", "Neptune", "Uranus", "Pluto",
    "Gravity", "Speed of light", "Big Bang", "Black hole", "Galaxy",
    "Milky Way", "Universe", "Atom", "DNA", "Evolution", "Climate change",
    "Photosynthesis", "Water", "Oxygen", "Carbon dioxide",

    # History
    "World War I", "World War II", "Ancient Rome", "Ancient Egypt",
    "French Revolution", "American Revolution", "Cold War",
    "Napoleon", "Albert Einstein", "Isaac Newton", "Galileo Galilei",
    "Alexander the Great", "Julius Caesar", "Cleopatra",
    "Renaissance", "Industrial Revolution",

    # Technology
    "Internet", "Computer", "Artificial intelligence", "Python (programming language)",
    "Telephone", "Television", "Electricity", "Steam engine",
    "Alexander Graham Bell", "Thomas Edison", "Nikola Tesla",
    "World Wide Web", "Social media", "Smartphone",

    # People
    "Lionel Messi", "Cristiano Ronaldo", "Michael Jordan",
    "Leonardo da Vinci", "William Shakespeare", "Mozart",
    "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela",
    "Elon Musk", "Steve Jobs", "Bill Gates",

    # Biology & Health
    "Human body", "Heart", "Brain", "Virus", "Bacteria",
    "Vaccine", "Antibiotic", "Cancer",

    # General Knowledge
    "Olympic Games", "FIFA World Cup", "Nobel Prize",
    "United Nations", "Democracy", "Communism",
    "Great Wall of China", "Pyramids of Giza", "Eiffel Tower",
    "Amazon River", "Nile", "Mount Everest", "Pacific Ocean",
]

OUTPUT_DIR = "data"
PASSAGE_FILE = os.path.join(OUTPUT_DIR, "wiki_passages.json")
INDEX_FILE = os.path.join(OUTPUT_DIR, "wiki_index.faiss")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200  # words per passage


def crawl_wikipedia(topics):
    """Crawl Wikipedia articles and split into passages."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="ConvoAI/1.0 (student project)",
        language="en",
    )

    passages = []
    seen_titles = set()

    for topic in topics:
        page = wiki.page(topic)
        if not page.exists():
            print(f"  SKIP: {topic} (not found)")
            continue
        if page.title in seen_titles:
            continue
        seen_titles.add(page.title)

        text = page.text
        if not text or len(text) < 100:
            continue

        # Split into chunks
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE // 2):  # 50% overlap
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            if len(chunk.split()) >= 20:  # minimum 20 words
                passages.append({
                    "text": chunk,
                    "title": page.title,
                    "source": f"Wikipedia: {page.title}",
                })

        print(f"  OK: {page.title} ({len(words)} words → {len(words) // (CHUNK_SIZE // 2)} passages)")

    return passages


def build_faiss_index(passages, model):
    """Encode passages and build FAISS index."""
    texts = [p["text"] for p in passages]
    print(f"\n  Encoding {len(texts)} passages...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (= cosine sim for normalized vectors)
    index.add(embeddings)

    return index


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading sentence-transformer...")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    print(f"\nCrawling {len(TOPICS)} Wikipedia topics...")
    passages = crawl_wikipedia(TOPICS)
    print(f"\nTotal passages: {len(passages)}")

    # Save passages
    with open(PASSAGE_FILE, "w") as f:
        json.dump(passages, f)
    print(f"Passages saved to {PASSAGE_FILE}")

    # Build and save FAISS index
    index = build_faiss_index(passages, model)
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index saved to {INDEX_FILE}")

    # Stats
    size_mb = os.path.getsize(INDEX_FILE) / (1024 * 1024)
    print(f"\nDone:")
    print(f"  {len(passages)} passages from {len(set(p['title'] for p in passages))} articles")
    print(f"  Index size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
