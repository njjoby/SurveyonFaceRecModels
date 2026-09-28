import os
import mxnet as mx
import cv2
import numpy as np

# The TensorFlow DeepFace implementation was removed in TensorFlow 2.16.
# DeepFace's PyTorch implementation supports the same pretrained model.
os.environ["DEEPFACE_BACKEND_ENGINE"] = "pytorch"
from deepface import DeepFace


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

REC_PATH = "CASIA-WebFace-400K/train.rec"
IDX_PATH = "CASIA-WebFace-400K/train.idx"

OUTPUT_DIR = "Embeddings Generated DeepFace/CASIA-WebFace-400K"

OUTPUT_EMBEDDINGS = os.path.join(
    OUTPUT_DIR,
    "deepface_embeddings.npy"
)

OUTPUT_LABELS = os.path.join(
    OUTPUT_DIR,
    "deepface_labels.npy"
)


# ============================================================
# 2. DEEPFACE MODEL
# ============================================================
# Preload the PyTorch-backed "DeepFace" model so an incompatibility or
# checkpoint problem fails once, before processing the RecordIO dataset.
# RecordIO faces are aligned, so face detection is skipped during extraction.

print("Loading DeepFace's pretrained model with the PyTorch backend")
DeepFace.build_model("DeepFace")
print("DeepFace model loaded.")
print()


# ============================================================
# 3. OPEN CASIA-WebFace-400K
# ============================================================

print("=" * 60)
print("Opening CASIA-WebFace-400K")
print("=" * 60)

record = mx.recordio.MXIndexedRecordIO(
    IDX_PATH,
    REC_PATH,
    "r"
)

number_of_records = len(record.keys)

print("Number of records:", number_of_records)
print()


# ============================================================
# 6. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 7. STORAGE FOR EMBEDDINGS AND LABELS
# ============================================================

all_embeddings = []
all_labels = []

successful = 0
failed = 0


# ============================================================
# 8. PROCESS EVERY IMAGE
# ============================================================

print("=" * 60)
print("Starting embedding extraction")
print("=" * 60)
print()


for i, index in enumerate(record.keys):

    try:

        # ----------------------------------------------------
        # READ RECORD
        # ----------------------------------------------------

        packed = record.read_idx(index)

        header, img_bytes = mx.recordio.unpack(packed)


        # ----------------------------------------------------
        # SKIP HEADER RECORDS
        # ----------------------------------------------------

        if header.flag == 1:
            continue


        # ----------------------------------------------------
        # READ LABEL
        # ----------------------------------------------------

        try:

            label = int(header.label)

        except (TypeError, ValueError):

            label_array = np.asarray(
                header.label
            ).reshape(-1)

            if len(label_array) == 0:

                print(
                    f"Skipping record {i}: "
                    f"invalid label = {header.label}"
                )

                failed += 1
                continue

            label = int(label_array[0])


        # ----------------------------------------------------
        # DECODE IMAGE
        # ----------------------------------------------------

        image_array = np.frombuffer(
            img_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        # ----------------------------------------------------
        # CHECK IMAGE DECODING
        # ----------------------------------------------------

        if image is None:

            print(
                f"Could not decode image "
                f"in record {i}"
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # CHECK IMAGE SIZE
        # ----------------------------------------------------

        height, width = image.shape[:2]

        if height != 112 or width != 112:

            print(
                f"Warning: record {i} "
                f"has size {width}x{height}"
            )


        # ====================================================
        # DEEPFACE EMBEDDING EXTRACTION
        # ====================================================
        # Input is an aligned BGR image from RecordIO. Skip detection and
        # alignment; DeepFace handles model-specific resizing and normalization.
        # ====================================================

        representation = DeepFace.represent(
            img_path=image,
            model_name="DeepFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            normalization="base",
            l2_normalize=False
        )
        embedding = representation[0]["embedding"]


        # ----------------------------------------------------
        # CONVERT EMBEDDING TO NUMPY
        # ----------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        embedding = embedding.reshape(-1)


        # ----------------------------------------------------
        # CHECK EMBEDDING
        # ----------------------------------------------------

        if embedding.size == 0:

            print(
                f"Empty embedding "
                f"for record {i}"
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # L2 NORMALIZATION
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:

            print(
                f"Zero embedding "
                f"for record {i}"
            )

            failed += 1
            continue


        embedding = (
            embedding / norm
        ).astype(np.float32)


        # ----------------------------------------------------
        # STORE EMBEDDING AND LABEL
        # ----------------------------------------------------

        all_embeddings.append(
            embedding
        )

        all_labels.append(
            label
        )

        successful += 1


        # ----------------------------------------------------
        # DISPLAY PROGRESS
        # ----------------------------------------------------

        if successful % 100 == 0:

            print(
                f"Processed: {successful:5d} / "
                f"{number_of_records} | "
                f"Failed: {failed}"
            )


    except Exception as e:

        print(
            f"Error processing record {i}: {e}"
        )

        failed += 1

        continue


# ============================================================
# 9. CHECK WHETHER EMBEDDINGS WERE GENERATED
# ============================================================

if len(all_embeddings) == 0:

    raise RuntimeError(
        "No embeddings were generated."
    )


# ============================================================
# 10. CONVERT TO NUMPY ARRAYS
# ============================================================

print()
print("=" * 60)
print("Converting embeddings to NumPy arrays")
print("=" * 60)

all_embeddings = np.stack(
    all_embeddings
).astype(np.float32)

all_labels = np.asarray(
    all_labels,
    dtype=np.int64
)


# ============================================================
# 11. SAVE EMBEDDINGS
# ============================================================

print()
print("=" * 60)
print("Saving embeddings")
print("=" * 60)

np.save(
    OUTPUT_EMBEDDINGS,
    all_embeddings
)

np.save(
    OUTPUT_LABELS,
    all_labels
)


# ============================================================
# 12. VERIFY SAVED FILES
# ============================================================

print()
print("=" * 60)
print("Embedding extraction completed")
print("=" * 60)

print(
    "Successful images :",
    successful
)

print(
    "Failed images     :",
    failed
)

print(
    "Embeddings shape  :",
    all_embeddings.shape
)

print(
    "Labels shape      :",
    all_labels.shape
)

print(
    "Embedding dimension:",
    all_embeddings.shape[1]
)

print(
    "Embedding dtype   :",
    all_embeddings.dtype
)

print()

print(
    "Embeddings saved to:"
)

print(
    OUTPUT_EMBEDDINGS
)

print()

print(
    "Labels saved to:"
)

print(
    OUTPUT_LABELS
)

print("=" * 60)
