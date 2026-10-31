from sklearn.datasets import fetch_20newsgroups
import json
from pathlib import Path

categories = [
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "sci.electronics",
]

dataset = fetch_20newsgroups(
    subset="train",
    categories=categories,
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=42,
)

records = []

for document_id, (text, target) in enumerate(
    zip(dataset.data, dataset.target)
):
    records.append(
        {
            "document_id": document_id,
            "category_id": int(target),
            "category": dataset.target_names[target],
            "text": text,
        }
    )

output_path = (
    Path(__file__).resolve().parent.parent
    / "Data"
    / "experiment2_newsgroups.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Dataset downloaded and saved successfully.")
print(f"Total documents: {len(records)}")
print(f"Output: {output_path}")

for category_id, category_name in enumerate(dataset.target_names):
    count = sum(
        1 for record in records
        if record["category_id"] == category_id
    )
    print(f"{category_name}: {count} documents")