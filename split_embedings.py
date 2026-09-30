import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(
    description="Split embeddings into reference and remaining sets, identity-wise."
)
parser.add_argument(
    "embeddings_path",
    help="Path to the embeddings .npy file (the matching labels file must be alongside it).",
)
args = parser.parse_args()

# ============================================================
# LOAD EMBEDDINGS AND LABELS
# ============================================================

embeddings_path = args.embeddings_path
labels_path = embeddings_path.replace("_embeddings.npy", "_labels.npy")

if labels_path == embeddings_path:
    parser.error("embeddings_path must end with '_embeddings.npy'")

embeddings = np.load(embeddings_path)

labels = np.load(labels_path)

print("Original embeddings shape:", embeddings.shape)
print("Original labels shape:", labels.shape)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = os.path.join(os.path.dirname(embeddings_path), "Split")

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# SPLIT SETTINGS
# ============================================================

# Number of embeddings from EACH identity
# to put in the first set

N_REFERENCE = 1


# ============================================================
# SPLIT DATA IDENTITY-WISE
# ============================================================

reference_embeddings = []
reference_labels = []

remaining_embeddings = []
remaining_labels = []


# Get all unique identities
identities = np.unique(labels)

print("\nNumber of identities:", len(identities))


for identity in identities:

    # --------------------------------------------------------
    # Find all embeddings belonging to this identity
    # --------------------------------------------------------

    indices = np.where(labels == identity)[0]

    # --------------------------------------------------------
    # Shuffle indices so the split is random
    # --------------------------------------------------------

    np.random.shuffle(indices)

    # --------------------------------------------------------
    # First N_REFERENCE embeddings
    # --------------------------------------------------------

    reference_indices = indices[:N_REFERENCE]

    # --------------------------------------------------------
    # Remaining embeddings
    # --------------------------------------------------------

    remaining_indices = indices[N_REFERENCE:]


    # --------------------------------------------------------
    # Store reference embeddings
    # --------------------------------------------------------

    reference_embeddings.extend(
        embeddings[reference_indices]
    )

    reference_labels.extend(
        labels[reference_indices]
    )


    # --------------------------------------------------------
    # Store remaining embeddings
    # --------------------------------------------------------

    remaining_embeddings.extend(
        embeddings[remaining_indices]
    )

    remaining_labels.extend(
        labels[remaining_indices]
    )


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

reference_embeddings = np.asarray(
    reference_embeddings,
    dtype=np.float32
)

reference_labels = np.asarray(
    reference_labels,
    dtype=np.int64
)

remaining_embeddings = np.asarray(
    remaining_embeddings,
    dtype=np.float32
)

remaining_labels = np.asarray(
    remaining_labels,
    dtype=np.int64
)


# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        output_dir,
        "reference_embeddings.npy"
    ),
    reference_embeddings
)

np.save(
    os.path.join(
        output_dir,
        "reference_labels.npy"
    ),
    reference_labels
)

np.save(
    os.path.join(
        output_dir,
        "remaining_embeddings.npy"
    ),
    remaining_embeddings
)

np.save(
    os.path.join(
        output_dir,
        "remaining_labels.npy"
    ),
    remaining_labels
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETED")
print("=" * 60)

print("\nReference set:")
print("Embeddings:", reference_embeddings.shape)
print("Labels    :", reference_labels.shape)

print("\nRemaining set:")
print("Embeddings:", remaining_embeddings.shape)
print("Labels    :", remaining_labels.shape)

print("\nFiles saved in:")
print(output_dir)

print("=" * 60)
