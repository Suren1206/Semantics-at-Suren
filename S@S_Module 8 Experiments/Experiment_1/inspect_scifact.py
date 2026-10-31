from beir.datasets.data_loader import GenericDataLoader

data_path = "datasets/scifact"

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")

print("Test corpus :", len(corpus))
print("Test queries:", len(queries))
print("Test qrels  :", len(qrels))