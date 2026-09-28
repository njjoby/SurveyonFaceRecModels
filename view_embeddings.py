import numpy as np

embeddings = np.load("Embeddings Generated/arcface_embeddings.npy")
labels = np.load("Embeddings Generated/arcface_labels.npy")

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)
print("Data type:", embeddings.dtype)

print("\nFirst embedding:")
print(embeddings[0])

print("\nFirst 10 dimensions:")
print(embeddings[0][:10])

print("\nFirst label:")
print(labels[0])

print("\nNorm of first embedding:")
print(np.linalg.norm(embeddings[0]))