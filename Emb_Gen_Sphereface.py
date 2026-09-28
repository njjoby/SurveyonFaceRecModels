"""Generate SphereFace embeddings for an MXNet RecordIO face dataset.

The pretrained Sphere20 ONNX model is downloaded automatically on first run.
No local model training is performed.
"""

import gc
import hashlib
import os
import tempfile
import urllib.request

import cv2
import mxnet as mx
import numpy as np
import onnxruntime as ort


# ============================================================
# Configuration
# ============================================================

REC_PATH = "CASIA-WebFace-400K/train.rec"
IDX_PATH = "CASIA-WebFace-400K/train.idx"
OUTPUT_DIR = "Embeddings Generated SphereFace/CASIA-WebFace-400K"
OUTPUT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "sphereface_embeddings.npy")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "sphereface_labels.npy")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sphere20.onnx")
MODEL_URLS = (
    "https://github.com/yakhyo/uniface/releases/download/weights/sphere20.onnx",
    "https://huggingface.co/yakhyo/uniface-weights/resolve/main/sphere20.onnx",
)
MODEL_SHA256 = "c02878cf658eb1861f580b7e7144b0d27cc29c440bcaa6a99d466d2854f14c9d"
EMBEDDING_SIZE = 512
INPUT_SIZE = (112, 112)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model():
    """Download the published Sphere20 model and verify its checksum."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.isfile(MODEL_PATH):
        if sha256_file(MODEL_PATH) == MODEL_SHA256:
            print("Using cached pretrained Sphere20 model:", MODEL_PATH)
            return
        print("Cached SphereFace model checksum is invalid; downloading a fresh copy.")
        os.remove(MODEL_PATH)

    last_error = None
    for url in MODEL_URLS:
        temporary_path = None
        try:
            print("Downloading pretrained Sphere20 model from:", url)
            with tempfile.NamedTemporaryFile(
                prefix="sphere20_", suffix=".onnx", dir=MODEL_DIR, delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
                request = urllib.request.Request(
                    url, headers={"User-Agent": "SphereFace-Embedding-Generator"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temporary_file.write(chunk)

            if sha256_file(temporary_path) != MODEL_SHA256:
                raise RuntimeError("Downloaded model failed SHA-256 verification")

            os.replace(temporary_path, MODEL_PATH)
            print("Pretrained Sphere20 model saved to:", MODEL_PATH)
            return
        except Exception as error:
            last_error = error
            print("Download source failed:", error)
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    raise RuntimeError(
        "Could not download a valid pretrained SphereFace model. "
        "Check your internet connection and try again."
    ) from last_error


def parse_label(raw_label, record_number):
    """Convert an MXNet RecordIO label to one integer identity label."""
    label_array = np.asarray(raw_label).reshape(-1)
    if label_array.size == 0:
        raise ValueError(f"Record {record_number} has no identity label")
    return int(label_array[0])


def main():
    download_model()

    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    print("ONNX Runtime providers:", providers)

    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    record = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, "r")
    number_of_records = len(record.keys)
    if number_of_records == 0:
        record.close()
        raise RuntimeError("The RecordIO dataset contains no records.")

    # Write to memory-mapped temporary arrays so all embeddings are not kept in RAM.
    temporary_embeddings = OUTPUT_EMBEDDINGS + ".working.npy"
    temporary_labels = OUTPUT_LABELS + ".working.npy"
    embeddings_map = np.lib.format.open_memmap(
        temporary_embeddings,
        mode="w+",
        dtype=np.float32,
        shape=(number_of_records, EMBEDDING_SIZE),
    )
    labels_map = np.lib.format.open_memmap(
        temporary_labels,
        mode="w+",
        dtype=np.int64,
        shape=(number_of_records,),
    )

    successful = 0
    failed = 0
    try:
        print(f"Processing {number_of_records} RecordIO entries")
        for record_number, key in enumerate(record.keys, start=1):
            packed = None
            header = None
            image_bytes = None
            image_array = None
            image = None
            blob = None
            embedding = None

            try:
                packed = record.read_idx(key)
                header, image_bytes = mx.recordio.unpack(packed)
                if header.flag == 1:
                    continue

                label = parse_label(header.label, record_number)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Could not decode image")

                # Input images are aligned CASIA faces. The model preprocessing
                # resizes to 112x112, converts BGR to RGB, and scales to [-1, 1].
                blob = cv2.dnn.blobFromImage(
                    image,
                    scalefactor=1.0 / 127.5,
                    size=INPUT_SIZE,
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True,
                )
                embedding = session.run([output_name], {input_name: blob})[0]
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding.size != EMBEDDING_SIZE:
                    raise ValueError(
                        f"Expected {EMBEDDING_SIZE} embedding values, got {embedding.size}"
                    )

                norm = np.linalg.norm(embedding)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("Embedding has an invalid or zero norm")

                embeddings_map[successful] = embedding / norm
                labels_map[successful] = label
                successful += 1

                if successful % 100 == 0:
                    print(
                        f"Processed: {successful} embeddings | "
                        f"Record: {record_number}/{number_of_records} | Failed: {failed}"
                    )
            except Exception as error:
                failed += 1
                print(f"Error processing record {record_number}: {error}")
            finally:
                del packed, header, image_bytes, image_array, image, blob, embedding
                if record_number % 500 == 0:
                    gc.collect()

        if successful == 0:
            raise RuntimeError("No SphereFace embeddings were generated.")

        embeddings_map.flush()
        labels_map.flush()
        np.save(OUTPUT_EMBEDDINGS, embeddings_map[:successful])
        np.save(OUTPUT_LABELS, labels_map[:successful])
    finally:
        record.close()
        del embeddings_map, labels_map
        gc.collect()
        for path in (temporary_embeddings, temporary_labels):
            if os.path.exists(path):
                os.remove(path)

    print("\nSphereFace embedding extraction completed")
    print("Successful images:", successful)
    print("Failed images:", failed)
    print("Embeddings shape:", (successful, EMBEDDING_SIZE))
    print("Labels shape:", (successful,))
    print("Embeddings saved to:", OUTPUT_EMBEDDINGS)
    print("Labels saved to:", OUTPUT_LABELS)


if __name__ == "__main__":
    main()
