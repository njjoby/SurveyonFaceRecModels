"""Generate face embeddings with a pretrained ViT face-recognition model.

The pretrained model is downloaded automatically from Hugging Face on first
run. No local training is performed.
"""

import gc
import os

import cv2
import mxnet as mx
import numpy as np
import timm
import torch
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

REC_PATH = "CASIA-WebFace-400K/train.rec"
IDX_PATH = "CASIA-WebFace-400K/train.idx"
OUTPUT_DIR = "Embeddings Generated ViT/CASIA-WebFace-400K"
OUTPUT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "vit_embeddings.npy")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "vit_labels.npy")

# ViT-Tiny trained for face recognition with ArcFace loss on MS1MV3.
# The checkpoint's architecture is a ViT with 512-dimensional output.
PRETRAINED_MODEL = "hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3"
INPUT_SIZE = (112, 112)
EMBEDDING_SIZE = 512


def parse_label(raw_label, record_number):
    label_array = np.asarray(raw_label).reshape(-1)
    if label_array.size == 0:
        raise ValueError(f"Record {record_number} has no identity label")
    return int(label_array[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading pretrained ViT face-recognition model:", PRETRAINED_MODEL)
    model = timm.create_model(PRETRAINED_MODEL, pretrained=True).to(device).eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    record = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, "r")
    number_of_records = len(record.keys)
    if number_of_records == 0:
        record.close()
        raise RuntimeError("The RecordIO dataset contains no records.")

    # Memory-map outputs to avoid retaining every embedding in RAM.
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
            packed = header = image_bytes = image_array = image = image_tensor = None
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

                # Model training uses RGB inputs normalized from [0, 1] to [-1, 1].
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
                image = image.astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image_tensor = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                ).unsqueeze(0).to(device)

                with torch.inference_mode():
                    embedding = model(image_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)

                embedding = embedding.reshape(-1)
                if embedding.size != EMBEDDING_SIZE or not np.isfinite(embedding).all():
                    raise ValueError(
                        f"Expected a finite {EMBEDDING_SIZE}-D embedding, "
                        f"got shape {embedding.shape}"
                    )

                embeddings_map[successful] = embedding
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
                del packed, header, image_bytes, image_array, image, image_tensor, embedding
                if record_number % 500 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        if successful == 0:
            raise RuntimeError("No ViT embeddings were generated.")

        embeddings_map.flush()
        labels_map.flush()
        np.save(OUTPUT_EMBEDDINGS, embeddings_map[:successful])
        np.save(OUTPUT_LABELS, labels_map[:successful])
    finally:
        record.close()
        del embeddings_map, labels_map
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for path in (temporary_embeddings, temporary_labels):
            if os.path.exists(path):
                os.remove(path)

    print("\nViT embedding extraction completed")
    print("Successful images:", successful)
    print("Failed images:", failed)
    print("Embeddings shape:", (successful, EMBEDDING_SIZE))
    print("Labels shape:", (successful,))
    print("Embeddings saved to:", OUTPUT_EMBEDDINGS)
    print("Labels saved to:", OUTPUT_LABELS)


if __name__ == "__main__":
    main()
