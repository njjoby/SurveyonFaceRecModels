import os
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

# Minimum number of images required for an identity
MIN_IMAGES_PER_IDENTITY = 10


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
invalid_records = 0


for count, index in enumerate(record.keys):

    # Record 0 is normally the dataset header
    if index == 0:
        continue

    # Read record
    try:
        packed = record.read_idx(index)
    except Exception as e:

        print(
            f"Skipping record {index}: "
            f"could not read record ({e})"
        )

        invalid_records += 1
        continue

    # Unpack record
    try:
        header, img = mx.recordio.unpack(packed)

    except Exception as e:

        print(
            f"Skipping record {index}: "
            f"invalid RecordIO data ({e})"
        )

        invalid_records += 1
        continue


    # ========================================================
    # Get identity label
    # ========================================================

    try:

        label = int(header.label)

        identity_records[label].append(index)

        total_images += 1

    except (TypeError, ValueError):

        print(
            f"Skipping record {index}: "
            f"invalid label = {header.label}"
        )

        invalid_records += 1
        continue


    # ========================================================
    # Progress
    # ========================================================

    if total_images % 50000 == 0:

        print(
            f"Processed {total_images:,} images..."
        )


# ============================================================
# Scanning completed
# ============================================================

print("\n==========================================")
print("Scanning completed")
print("==========================================")

print(
    "Total valid images:",
    f"{total_images:,}"
)

print(
    "Total identities found:",
    len(identity_records)
)

print(
    "Invalid records:",
    invalid_records
)


# ============================================================
# Find identities having at least MIN_IMAGES_PER_IDENTITY
# ============================================================

print(
    f"\nFinding identities with at least "
    f"{MIN_IMAGES_PER_IDENTITY} images..."
)


eligible_identities = [
    identity
    for identity, indices in identity_records.items()
    if len(indices) >= MIN_IMAGES_PER_IDENTITY
]


# ============================================================
# Display statistics
# ============================================================

print(
    "\nIdentities satisfying minimum requirement:",
    len(eligible_identities)
)


# ============================================================
# Check whether any identities are eligible
# ============================================================

if len(eligible_identities) == 0:

    raise RuntimeError(
        f"No identities have at least "
        f"{MIN_IMAGES_PER_IDENTITY} images."
    )


# ============================================================
# IMPORTANT
#
# We keep ALL eligible identities.
#
# There is NO upper limit on:
#
#   - Number of identities
#   - Number of images per identity
# ============================================================

selected_identities = sorted(eligible_identities)


print(
    "\nSelected identities:",
    len(selected_identities)
)


# ============================================================
# Prepare selected records
# ============================================================

selected_records = []


for new_label, original_label in enumerate(
    selected_identities
):

    # Get ALL images belonging to this identity
    available_images = identity_records[original_label]


    # --------------------------------------------------------
    # DO NOT sample.
    #
    # We keep every image.
    # --------------------------------------------------------

    for index in available_images:

        selected_records.append(
            (
                index,
                original_label,
                new_label
            )
        )


# ============================================================
# Calculate statistics
# ============================================================

total_selected_images = len(selected_records)

min_images = min(
    len(identity_records[identity])
    for identity in selected_identities
)

max_images = max(
    len(identity_records[identity])
    for identity in selected_identities
)

average_images = (
    total_selected_images /
    len(selected_identities)
)


# ============================================================
# Display selection statistics
# ============================================================

print("\n==========================================")
print("Dataset selection statistics")
print("==========================================")

print(
    "Number of identities:",
    len(selected_identities)
)

print(
    "Minimum images per identity:",
    min_images
)

print(
    "Maximum images per identity:",
    max_images
)

print(
    "Average images per identity:",
    f"{average_images:.2f}"
)

print(
    "Total selected images:",
    f"{total_selected_images:,}"
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


    # ========================================================
    # Read original record
    # ========================================================

    try:

        packed = record.read_idx(
            original_index
        )

    except Exception as e:

        print(
            f"\nERROR reading original record "
            f"{original_index}: {e}"
        )

        continue


    # ========================================================
    # Unpack original record
    # ========================================================

    try:

        header, img = mx.recordio.unpack(
            packed
        )

    except Exception as e:

        print(
            f"\nERROR unpacking record "
            f"{original_index}: {e}"
        )

        continue


    # ========================================================
    # Create new header
    #
    # The original identity label is replaced with
    # a continuous label:
    #
    # 0, 1, 2, 3, ...
    # ========================================================

    new_header = mx.recordio.IRHeader(
        header.flag,
        new_label,
        new_index,
        header.id2
    )


    # ========================================================
    # Pack image with new label
    # ========================================================

    new_packed = mx.recordio.pack(
        new_header,
        img
    )


    # ========================================================
    # Write to new RecordIO
    # ========================================================

    output_record.write_idx(
        new_index,
        new_packed
    )


    # ========================================================
    # Progress
    # ========================================================

    if (new_index + 1) % 1000 == 0:

        print(
            f"Written "
            f"{new_index + 1:,} / "
            f"{total_selected_images:,} images..."
        )


# ============================================================
# Close output RecordIO
# ============================================================

output_record.close()


# ============================================================
# Close input RecordIO
# ============================================================

record.close()


# ============================================================
# Final information
# ============================================================

print("\n")
print("==========================================")
print("CASIA-WebFace subset creation completed!")
print("==========================================")

print("\nOutput directory:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nFiles created:")

print(
    "  ",
    OUTPUT_REC
)

print(
    "  ",
    OUTPUT_IDX
)

print("\nDataset statistics:")

print(
    "  Identities:",
    len(selected_identities)
)

print(
    "  Minimum images per identity:",
    min_images
)

print(
    "  Maximum images per identity:",
    max_images
)

print(
    "  Average images per identity:",
    f"{average_images:.2f}"
)

print(
    "  Total images:",
    f"{total_selected_images:,}"
)

print("\nSelection rule:")

print(
    f"  Identity included if it has "
    f">= {MIN_IMAGES_PER_IDENTITY} images"
)

print(
    "  ALL images of eligible identities are included."
)

print(
    "  There is NO upper limit on images per identity."
)

print("\nDataset structure:")

print(
    f"{OUTPUT_DIR}/"
)

print(
    "├── train.rec"
)

print(
    "└── train.idx"
)

print("\n==========================================")
print("Done!")
print("==========================================")