from datasets import load_dataset
df = load_dataset("wikitablequestions")
print(df)

print(df["train"][0])
print(df["train"][1])