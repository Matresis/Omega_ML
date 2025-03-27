import pickle as pc

with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

# Check if 'Ford' is in the encoding
print("Is Ford in encoding?", 'Ford' in brand_encoding)
