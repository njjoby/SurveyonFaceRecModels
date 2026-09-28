import numpy as np
import sys
import gc


# ============================================================
# SAVE ALL PRINT OUTPUT TO TERMINAL + out.txt
# ============================================================

class Tee:

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


output_file = open(
    "out.txt",
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(
    sys.__stdout__,
    output_file
)


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

print()
print("Starting L2 normalization...")

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

print("L2 normalization completed.")
gc.collect()


# ============================================================
# COSINE SIMILARITY
# ============================================================

print()
print("Starting cosine similarity calculation...")

similarity_matrix = np.dot(
    query_embeddings,
    gallery_embeddings.T
)

# These embeddings are no longer needed after the similarity matrix is built.
# Keep their counts for the summary printed later.
total_queries = len(query_embeddings)
gallery_count = len(gallery_embeddings)
del query_embeddings, gallery_embeddings
gc.collect()

print("Cosine similarity calculation completed.")

print()
print(
    "Similarity matrix:",
    similarity_matrix.shape
)


# ============================================================
# FIND TOP-3 MATCHES
# ============================================================

print()
print("Finding Top-3 matches...")

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

del similarity_matrix
gc.collect()


print("Top-3 matching completed.")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 90)
print("TOP-3 BEST MATCHES FOR EACH QUERY")
print("=" * 90)


for i in range(
    total_queries
):

    print()

    print(
        f"Query {i} | "
        f"True Identity: {query_labels[i]}"
    )


    for rank in range(
        TOP_K
    ):

        gallery_index = (
            top3_indices[i, rank]
        )

        similarity = (
            top3_scores[i, rank]
        )

        matched_label = (
            top3_labels[i, rank]
        )


        print(
            f"   Rank {rank + 1}: "
            f"Gallery Index = "
            f"{gallery_index:3d} | "
            f"Identity = "
            f"{matched_label:3d} | "
            f"Similarity = "
            f"{similarity:.4f}"
        )


# Displaying matches is the last use of their indices and scores.
del top3_indices, top3_scores
gc.collect()


# ============================================================
# TOP-1 ACCURACY
# ============================================================

print()
print("Calculating Top-1 accuracy...")

top1_predictions = (
    top3_labels[:, 0]
)

top1_correct = (
    top1_predictions ==
    query_labels
)

top1_accuracy = (
    np.mean(top1_correct) *
    100
)

print("Top-1 accuracy calculation completed.")
del top1_predictions, top1_correct
gc.collect()


# ============================================================
# TOP-3 ACCURACY
# ============================================================

print("Calculating Top-3 accuracy...")


# Check whether the correct identity occurs
# anywhere among the top 3 results.

top3_correct = np.any(
    top3_labels ==
    query_labels[:, None],
    axis=1
)


top3_accuracy = (
    np.mean(top3_correct) *
    100
)

print("Top-3 accuracy calculation completed.")
del top3_correct, query_labels, gallery_labels, top3_labels
gc.collect()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("RETRIEVAL SUMMARY")
print("=" * 90)


print(
    "Total queries       :",
    total_queries
)


print(
    "Gallery embeddings  :",
    gallery_count
)


print(
    f"Top-1 Accuracy      : "
    f"{top1_accuracy:.2f}%"
)


print(
    f"Top-3 Accuracy      : "
    f"{top3_accuracy:.2f}%"
)


print("=" * 90)
del total_queries, gallery_count
gc.collect()


# ============================================================
# CLOSE OUTPUT FILE SAFELY
# ============================================================

sys.stdout = sys.__stdout__

output_file.close()

print()
print("Results saved to out.txt")
