"""Download full FEVER dataset — all topics, all claims."""
import json
from datasets import load_dataset

def main():
    print("Downloading FEVER dataset from HuggingFace...")
    ds = load_dataset("copenlu/fever_gold_evidence", split="train")
    print(f"  Total records: {len(ds)}")

    facts = []
    seen = set()

    for row in ds:
        claim = row.get("claim", "").strip()
        label = row.get("label", "")

        if not claim or claim in seen:
            continue
        seen.add(claim)

        if label == "SUPPORTS":
            facts.append({"claim": claim, "verdict": True, "source": "FEVER/Wikipedia", "category": "general"})
        elif label == "REFUTES":
            facts.append({"claim": claim, "verdict": False, "source": "FEVER/Wikipedia", "category": "general",
                          "correction": "This claim has been verified as false based on Wikipedia evidence."})

    true_c = sum(1 for f in facts if f["verdict"])
    false_c = sum(1 for f in facts if not f["verdict"])
    print(f"\n  Total claims: {len(facts)}")
    print(f"  True (SUPPORTS): {true_c}")
    print(f"  False (REFUTES): {false_c}")

    data = {
        "domain": "general_knowledge",
        "description": "Full FEVER dataset — Wikipedia-verified claims across all topics.",
        "facts": facts,
    }

    with open("data/facts.json", "w") as f:
        json.dump(data, f)
    print(f"  Saved to data/facts.json")

if __name__ == "__main__":
    main()
