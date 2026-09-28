import os
import random
import mxnet as mx
from collections import defaultdict


# ============================================================
# Configuration
# ============================================================

SOURCE_REC = "CASIA-WebFace/train.rec"
SOURCE_IDX = "CASIA-WebFace/train.idx"

OUTPUT_DIR = "CASIA-WebFace-MAX"
OUTPUT_REC = os.path.join(OUTPUT_DIR, "train.rec")
OUTPUT_IDX = os.path.join(OUTPUT_DIR, "train.idx")

NUM_IDENTITIES = 10500
IMAGES_PER_IDENTITY = 30

RANDOM_SEED = 42


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)


# ============================================================
# Open original CASIA-WebFace dataset
# ============================================================

print("Opening CASIA-WebFace...")

record = mx.recordio.MXIndexedRecordIO(
    SOURCE_IDX,
    SOURCE_REC,
    "r"
)

print("Total records:", len(record.keys))


# ============================================================
# Group records by identity
# ============================================================

print("\nScanning dataset and grouping images by identity...")

identity_records = defaultdict(list)

total_images = 0

for count, index in enumerate(record.keys):

    # Record 0 is normally the dataset header.
    if index == 0:
        continue

    packed = record.read_idx(index)

    try:
        header, img = mx.recordio.unpack(packed)
    except Exception:
        print("Skipping invalid record:", index)
        continue

    # Identity label
    
    try:
        label = int(header.label)
        identity_records[label].append(index)
    except (TypeError, ValueError):
        print(
            f"Skipping record {index}: "
            f"invalid label = {header.label}"
                )
        continue

   

    total_images += 1

    # Progress
    if total_images % 50000 == 0:
        print(
            f"Processed {total_images:,} images..."
        )


print("\nFinished scanning.")

print("Total images found:", total_images)
print("Total identities found:", len(identity_records))


# ============================================================
# Find identities with at least 20 images
# ============================================================

eligible_identities = [
    identity
    for identity, indices in identity_records.items()
    if len(indices) >= IMAGES_PER_IDENTITY
]


print(
    "Identities with at least",
    IMAGES_PER_IDENTITY,
    "images:",
    len(eligible_identities)
)


# ============================================================
# Check whether enough identities are available
# ============================================================

if len(eligible_identities) < NUM_IDENTITIES:

    raise RuntimeError(
        f"Only {len(eligible_identities)} identities have "
        f"at least {IMAGES_PER_IDENTITY} images. "
        f"Cannot create {NUM_IDENTITIES} identities."
    )


# ============================================================
# Randomly select identities
# ============================================================

selected_identities = random.sample(
    eligible_identities,
    NUM_IDENTITIES
)


print("\nSelected identities:", len(selected_identities))


# ============================================================
# Select images for each identity
# ============================================================

selected_records = []

for new_label, original_label in enumerate(selected_identities):

    available_images = identity_records[original_label]

    selected_images = random.sample(
        available_images,
        IMAGES_PER_IDENTITY
    )

    for index in selected_images:

        selected_records.append(
            (
                index,
                original_label,
                new_label
            )
        )


# Shuffle the complete dataset

random.shuffle(selected_records)


print(
    "Total selected images:",
    len(selected_records)
)


# ============================================================
# Create new RecordIO files
# ============================================================

print("\nCreating output RecordIO dataset...")

output_record = mx.recordio.MXIndexedRecordIO(
    OUTPUT_IDX,
    OUTPUT_REC,
    "w"
)


# ============================================================
# Write selected images
# ============================================================

for new_index, (
    original_index,
    original_label,
    new_label
) in enumerate(selected_records):

    # Read original record
    packed = record.read_idx(original_index)

    # Unpack
    header, img = mx.recordio.unpack(packed)

    # Create new header with remapped label
    new_header = mx.recordio.IRHeader(
        header.flag,
        new_label,
        new_index,
        header.id2
    )

    # Pack image with new label
    new_packed = mx.recordio.pack(
        new_header,
        img
    )

    # Write to new RecordIO
    output_record.write_idx(
        new_index,
        new_packed
    )

    if (new_index + 1) % 1000 == 0:

        print(
            f"Written {new_index + 1:,} / "
            f"{len(selected_records):,} images..."
        )


# ============================================================
# Close output
# ============================================================

output_record.close()


# ============================================================
# Final information
# ============================================================

print("\n==========================================")
print("CASIA-WebFace 10K creation completed!")
print("==========================================")

print("Output directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

print("  ", OUTPUT_REC)
print("  ", OUTPUT_IDX)

print("\nIdentities:", NUM_IDENTITIES)
print("Images per identity:", IMAGES_PER_IDENTITY)
print("Total images:", len(selected_records))

print("\nRandom seed:", RANDOM_SEED)

print("\nDataset structure:")
print("CASIA-WebFace-10K/")
print("├── train.rec")
print("└── train.idx")