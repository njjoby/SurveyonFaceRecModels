import numpy as np


# ============================================================
# LOAD QUERY EMBEDDINGS
# ============================================================

query_embeddings = np.load(
    "Embeddings Generated/CASIA-WebFace-400K/Split/reference_embeddings.npy"
)

query_labels = np.load(
    "Embeddings Generated/CASIA-WebFace-400K/Split/reference_labels.npy"
)


# ============================================================
# LOAD GALLERY EMBEDDINGS
# ============================================================

gallery_embeddings = np.load(
    "Embeddings Generated/CASIA-WebFace-400K/Split/remaining_embeddings.npy"
)

gallery_labels = np.load(
    "Embeddings Generated/CASIA-WebFace-400K/Split/remaining_labels.npy"
)


# ============================================================
# DISPLAY SHAPES
# ============================================================

print("Query embeddings   :", query_embeddings.shape)
print("Query labels       :", query_labels.shape)

print("Gallery embeddings :", gallery_embeddings.shape)
print("Gallery labels     :", gallery_labels.shape)


# ============================================================
# L2 NORMALIZATION
# ============================================================

query_embeddings = (
    query_embeddings /
    np.linalg.norm(
        query_embeddings,
        axis=1,
        keepdims=True
    )
)

gallery_embeddings = (
    gallery_embeddings /
    np.linalg.norm(
        gallery_embeddings,
        axis=1,
        keepdims=True
    )
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity_matrix = np.dot(
    query_embeddings,
    gallery_embeddings.T
)

print()
print("Similarity matrix:", similarity_matrix.shape)


# ============================================================
# FIND TOP-3 MATCHES
# ============================================================

TOP_K = 3

# Get indices of the top 3 similarities
top3_indices = np.argsort(
    similarity_matrix,
    axis=1
)[:, -TOP_K:]


# Reverse order so highest similarity comes first
top3_indices = np.flip(
    top3_indices,
    axis=1
)


# Get corresponding similarity scores
top3_scores = np.take_along_axis(
    similarity_matrix,
    top3_indices,
    axis=1
)


# Get corresponding identities
top3_labels = gallery_labels[
    top3_indices
]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 90)
print("TOP-3 BEST MATCHES FOR EACH QUERY")
print("=" * 90)

for i in range(len(query_embeddings)):

    print()
    print(
        f"Query {i} | "
        f"True Identity: {query_labels[i]}"
    )

    for rank in range(TOP_K):

        gallery_index = top3_indices[i, rank]

        similarity = top3_scores[i, rank]

        matched_label = top3_labels[i, rank]

        print(
            f"   Rank {rank + 1}: "
            f"Gallery Index = {gallery_index:3d} | "
            f"Identity = {matched_label:3d} | "
            f"Similarity = {similarity:.4f}"
        )


# ============================================================
# TOP-1 ACCURACY
# ============================================================

top1_predictions = top3_labels[:, 0]

top1_correct = (
    top1_predictions == query_labels
)

top1_accuracy = (
    np.mean(top1_correct) * 100
)


# ============================================================
# TOP-3 ACCURACY
# ============================================================

# Check whether the correct identity occurs
# anywhere among the top 3 results.

top3_correct = np.any(
    top3_labels == query_labels[:, None],
    axis=1
)

top3_accuracy = (
    np.mean(top3_correct) * 100
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("RETRIEVAL SUMMARY")
print("=" * 90)

print(
    "Total queries       :",
    len(query_embeddings)
)

print(
    "Gallery embeddings  :",
    len(gallery_embeddings)
)

print(
    f"Top-1 Accuracy      : {top1_accuracy:.2f}%"
)

print(
    f"Top-3 Accuracy      : {top3_accuracy:.2f}%"
)

print("=" * 90)