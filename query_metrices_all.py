import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import gc

START_TIME = time.perf_counter()


# ============================================================
# Configuration
# ============================================================

REFERENCE_EMBEDDINGS = "Embeddings Generated/CASIA-WebFace-400K/Split/reference_embeddings.npy"
REFERENCE_LABELS = "Embeddings Generated/CASIA-WebFace-400K/Split/reference_labels.npy"

GALLERY_EMBEDDINGS = "Embeddings Generated/CASIA-WebFace-400K/Split/remaining_embeddings.npy"
GALLERY_LABELS = "Embeddings Generated/CASIA-WebFace-400K/Split/remaining_labels.npy"

CMC_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/CMC_Curve.png"
TAR_FAR_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/TAR_FAR_Curve.png"
TAR_FAR_VALUES_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/TAR_FAR_Values.txt"
RECALL_AT_K_PLOT_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/Recall_at_K_Curve.png"
RECALL_AT_K_VALUES_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/Recall_at_K_Values.txt"
PRECISION_AT_K_PLOT_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/Precision_at_K_Curve.png"
PRECISION_AT_K_VALUES_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/Precision_at_K_Values.txt"

OUTPUT_FILE = "Embeddings Generated/CASIA-WebFace-400K/Split/query_metrics_results.txt"

MAX_RANK = 20


# ============================================================
# Output file
# ============================================================



log_file = open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
)


# ============================================================
# Tee class
# ============================================================
# Prints simultaneously to:
#   1. Terminal
#   2. Text file
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


# Redirect stdout
original_stdout = sys.stdout

sys.stdout = Tee(
    original_stdout,
    log_file
)

LAST_SECTION_TIME = START_TIME


def report_section_time(section_name):
    """Print time spent in this section and total time since startup."""
    global LAST_SECTION_TIME
    now = time.perf_counter()
    section_elapsed = now - LAST_SECTION_TIME
    total_elapsed = now - START_TIME
    LAST_SECTION_TIME = now

    def format_duration(duration):
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(
        f"\n[{section_name}] Section time: "
        f"{format_duration(section_elapsed)} | "
        f"Total elapsed: {format_duration(total_elapsed)} (hh:mm:ss)"
    )


# ============================================================
# Load embeddings and labels
# ============================================================

print("Loading embeddings...")

query_embeddings = np.load(
    REFERENCE_EMBEDDINGS
)

query_labels = np.load(
    REFERENCE_LABELS
)

gallery_embeddings = np.load(
    GALLERY_EMBEDDINGS
)

gallery_labels = np.load(
    GALLERY_LABELS
)

report_section_time("Load embeddings and labels")


print("\nDataset information:")

print(
    "Query embeddings:",
    query_embeddings.shape
)

print(
    "Query labels:",
    query_labels.shape
)

print(
    "Gallery embeddings:",
    gallery_embeddings.shape
)

print(
    "Gallery labels:",
    gallery_labels.shape
)


# ============================================================
# Normalize embeddings
# ============================================================
# This makes dot product equivalent to cosine similarity.

print("\nNormalizing embeddings...")

query_norms = np.linalg.norm(
    query_embeddings,
    axis=1,
    keepdims=True
)

gallery_norms = np.linalg.norm(
    gallery_embeddings,
    axis=1,
    keepdims=True
)

query_embeddings = (
    query_embeddings /
    (query_norms + 1e-12)
)

gallery_embeddings = (
    gallery_embeddings /
    (gallery_norms + 1e-12)
)

# Norm arrays are temporary inputs to normalization.
del query_norms, gallery_norms
gc.collect()

report_section_time("Normalize embeddings")


# ============================================================
# Calculate similarity matrix
# ============================================================

print("\nCalculating cosine similarity...")

similarity_matrix = np.dot(
    query_embeddings,
    gallery_embeddings.T
)

# The normalized embeddings have served their last purpose.
del query_embeddings, gallery_embeddings
gc.collect()


print(
    "Similarity matrix:",
    similarity_matrix.shape
)

report_section_time("Calculate cosine similarity")


# ============================================================
# Calculate Rank-1 to Rank-20
# ============================================================

print(
    f"\nCalculating Rank-1 to Rank-{MAX_RANK}..."
)


# Sort gallery indices according to similarity
# Highest similarity first

sorted_indices = np.argsort(
    -similarity_matrix,
    axis=1
)

report_section_time("Sort gallery by similarity")


# ============================================================
# Calculate MRR and mean Average Precision
# ============================================================

print("\nCalculating MRR and mAP...")

reciprocal_ranks = []
average_precisions = []

for query_index, query_label in enumerate(query_labels):

    ranked_labels = gallery_labels[sorted_indices[query_index]]
    relevant_positions = np.flatnonzero(ranked_labels == query_label) + 1

    if len(relevant_positions) == 0:
        reciprocal_ranks.append(0.0)
        average_precisions.append(0.0)
        continue

    reciprocal_ranks.append(1.0 / relevant_positions[0])

    precision_at_relevant = (
        np.arange(1, len(relevant_positions) + 1) /
        relevant_positions
    )
    average_precisions.append(np.mean(precision_at_relevant))

mrr = float(np.mean(reciprocal_ranks))
mean_average_precision = float(np.mean(average_precisions))

print(f"MRR: {mrr:.6f}")
print(f"mAP: {mean_average_precision:.6f}")

report_section_time("Calculate MRR and mAP")
del reciprocal_ranks, average_precisions
del ranked_labels, relevant_positions, precision_at_relevant
gc.collect()


# ============================================================
# Build a binned TAR@FAR curve
# ============================================================
# Cosine scores are in [-1, 1]. Histograms avoid storing separate
# copies of all genuine and impostor pair scores.

print("\nCalculating TAR@FAR curve...")

TAR_FAR_BINS = 2000
score_edges = np.linspace(-1.0, 1.0, TAR_FAR_BINS + 1)
genuine_histogram = np.zeros(TAR_FAR_BINS, dtype=np.int64)
impostor_histogram = np.zeros(TAR_FAR_BINS, dtype=np.int64)
tar_values = far_values = threshold_values = visible = None

for query_index, query_label in enumerate(query_labels):
    is_genuine = gallery_labels == query_label
    row_scores = similarity_matrix[query_index]
    genuine_histogram += np.histogram(
        row_scores[is_genuine], bins=score_edges
    )[0]
    impostor_histogram += np.histogram(
        row_scores[~is_genuine], bins=score_edges
    )[0]
    del is_genuine, row_scores

total_genuine = genuine_histogram.sum()
total_impostor = impostor_histogram.sum()

if total_genuine == 0 or total_impostor == 0:
    print("Cannot calculate TAR@FAR: need both genuine and impostor pairs.")
else:
    tar_values = np.cumsum(genuine_histogram[::-1]) / total_genuine
    far_values = np.cumsum(impostor_histogram[::-1]) / total_impostor
    threshold_values = score_edges[:-1][::-1]

    np.savetxt(
        TAR_FAR_VALUES_FILE,
        np.column_stack((threshold_values, far_values, tar_values)),
        delimiter="\t",
        header="threshold\tFAR\tTAR",
        comments="",
        fmt="%.10g"
    )
    print("TAR@FAR values saved as:")
    print(os.path.abspath(TAR_FAR_VALUES_FILE))

    # A logarithmic FAR axis cannot display zero.
    visible = far_values > 0
    plt.figure(figsize=(8, 6))
    plt.plot(far_values[visible], tar_values[visible], linewidth=2)
    plt.xscale("log")
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("True Accept Rate (TAR)")
    plt.title("TAR@FAR Curve")
    plt.xlim(max(float(far_values[visible].min()), 1e-7), 1.0)
    plt.ylim(0, 1)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(TAR_FAR_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print("TAR@FAR curve saved as:")
    print(os.path.abspath(TAR_FAR_FILE))

report_section_time("Calculate and save TAR@FAR")

# Similarities and TAR/FAR work arrays are no longer needed. Keep sorted
# indices for the CMC and Recall/Precision@K sections.
del similarity_matrix, score_edges, genuine_histogram, impostor_histogram
del tar_values, far_values, threshold_values, visible
del total_genuine, total_impostor
gc.collect()


# ============================================================
# Calculate CMC
# ============================================================

rank_accuracies = []


for rank in range(
    1,
    MAX_RANK + 1
):

    correct = 0

    for query_index in range(
        len(query_labels)
    ):

        # Top-K gallery results

        top_k_indices = sorted_indices[
            query_index,
            :rank
        ]

        # Corresponding labels

        top_k_labels = gallery_labels[
            top_k_indices
        ]

        # Check whether the correct identity
        # occurs anywhere in Top-K

        if query_labels[
            query_index
        ] in top_k_labels:

            correct += 1


    accuracy = (
        correct /
        len(query_labels)
    ) * 100


    rank_accuracies.append(
        accuracy
    )


    print(
        f"Rank-{rank}: "
        f"{accuracy:.2f}%"
    )

report_section_time("Calculate CMC ranks")
del top_k_indices, top_k_labels
gc.collect()


# ============================================================
# Print CMC table
# ============================================================

print(
    "\n=========================================="
)

print(
    "CMC Results"
)

print(
    "=========================================="
)

print(
    f"{'Rank':<10}"
    f"{'Accuracy':<15}"
)

print(
    "------------------------------------------"
)


for rank, accuracy in enumerate(
    rank_accuracies,
    start=1
):

    print(
        f"{rank:<10}"
        f"{accuracy:.2f}%"
    )


print(
    "=========================================="
)

report_section_time("Print CMC table")


# ============================================================
# Calculate Recall@K and Precision@K
# ============================================================

max_k = min(MAX_RANK, len(gallery_labels))
k_values = np.arange(1, max_k + 1)
precision_sums = np.zeros(max_k, dtype=np.float64)
recall_sums = np.zeros(max_k, dtype=np.float64)

for query_index, query_label in enumerate(query_labels):
    top_k_labels = gallery_labels[sorted_indices[query_index, :max_k]]
    relevant_in_top_k = (top_k_labels == query_label).astype(np.int64)
    cumulative_relevant = np.cumsum(relevant_in_top_k)
    total_relevant = np.count_nonzero(gallery_labels == query_label)

    precision_sums += cumulative_relevant / k_values
    if total_relevant > 0:
        recall_sums += cumulative_relevant / total_relevant

precision_at_k = precision_sums / len(query_labels)
recall_at_k = recall_sums / len(query_labels)

np.savetxt(
    PRECISION_AT_K_VALUES_FILE,
    np.column_stack((k_values, precision_at_k)),
    delimiter="\t",
    header="K\tPrecision_at_K",
    comments="",
    fmt=["%d", "%.10g"]
)

np.savetxt(
    RECALL_AT_K_VALUES_FILE,
    np.column_stack((k_values, recall_at_k)),
    delimiter="\t",
    header="K\tRecall_at_K",
    comments="",
    fmt=["%d", "%.10g"]
)

for values, title, ylabel, plot_file, values_file in (
    (precision_at_k, "Precision@K Curve", "Precision@K", PRECISION_AT_K_PLOT_FILE, PRECISION_AT_K_VALUES_FILE),
    (recall_at_k, "Recall@K Curve", "Recall@K", RECALL_AT_K_PLOT_FILE, RECALL_AT_K_VALUES_FILE),
):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, values, marker="o", linewidth=2)
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(k_values)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n{title} saved as:")
    print(os.path.abspath(plot_file))
    print(f"{title} values saved as:")
    print(os.path.abspath(values_file))

report_section_time("Calculate and save Recall@K and Precision@K")

# Retrieval arrays and labels are no longer needed after these metrics are saved.
del sorted_indices, query_labels, gallery_labels
del precision_sums, recall_sums, relevant_in_top_k, cumulative_relevant
del top_k_labels, total_relevant
gc.collect()


# ============================================================
# Plot CMC Curve
# ============================================================

ranks = np.arange(
    1,
    MAX_RANK + 1
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    ranks,
    rank_accuracies,
    marker="o",
    linewidth=2
)


plt.xlabel(
    "Rank"
)

plt.ylabel(
    "Identification Accuracy (%)"
)

plt.title(
    "CMC Curve"
)


plt.xticks(
    ranks
)

plt.ylim(
    0,
    100
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.6
)

plt.tight_layout()


# ============================================================
# Save CMC plot
# ============================================================



plt.savefig(
    CMC_FILE,
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print(
    "\nCMC curve saved as:"
)

print(
    os.path.abspath(CMC_FILE)
)

report_section_time("Plot and save CMC curve")
del ranks, rank_accuracies
del k_values, precision_at_k, recall_at_k, max_k
gc.collect()


# ============================================================
# Finish logging
# ============================================================

print(
    "\nResults saved to:"
)

print(
    os.path.abspath(OUTPUT_FILE)
)

elapsed_seconds = time.perf_counter() - START_TIME
hours, remainder = divmod(int(elapsed_seconds), 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nElapsed time: {hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss)")


# Restore stdout
sys.stdout = original_stdout

# Close log file
log_file.close()
