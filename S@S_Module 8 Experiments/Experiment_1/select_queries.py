import random
from beir.datasets.data_loader import GenericDataLoader

data_path = "datasets/scifact"

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")

random.seed(42)

selected_ids = random.sample(list(queries.keys()), 10)

print("\nSelected test queries:\n")

for query_id in selected_ids:
    print(f"{query_id}: {queries[query_id]}")