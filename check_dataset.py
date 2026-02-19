import pandas as pd

df = pd.read_csv("dataset/fake news dataset.csv", encoding="latin1")

print("Columns:\n", df.columns)
print("\nUnique Label Values:\n", df["labels"].unique())
print("\nDataset Shape:", df.shape)
