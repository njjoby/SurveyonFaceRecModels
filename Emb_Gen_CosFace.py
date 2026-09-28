import os

import cv2
import mxnet as mx
import numpy as np
import torch
from torchvision import transforms

from cosface_model import CosFaceEmbeddingModel


# ============================================================
# Configuration
# ============================================================

REC_PATH = "CASIA-WebFace-400K/train.rec"
IDX_PATH = "CASIA-WebFace-400K/train.idx"
OUTPUT_DIR = "Embeddings Generated CosFace/CASIA-WebFace-400K"
OUTPUT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "cosface_embeddings.npy")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "cosface_labels.npy")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CosFaceEmbeddingModel().to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

record = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, "r")
all_embeddings = []
all_labels = []
failed = 0

for record_number, key in enumerate(record.keys, start=1):
    try:
        packed = record.read_idx(key)
        header, image_bytes = mx.recordio.unpack(packed)
        if header.flag == 1:
            continue

        raw_label = np.asarray(header.label).reshape(-1)
        if raw_label.size == 0:
            raise ValueError("Record has no identity label")
        label = int(raw_label[0])

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("Could not decode image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image_tensor)[0].cpu().numpy().astype(np.float32)

        all_embeddings.append(embedding)
        all_labels.append(label)
        del image_tensor

        if len(all_embeddings) % 100 == 0:
            print(
                f"Processed: {len(all_embeddings)} / {len(record.keys)} | "
                f"Failed: {failed}"
            )
    except Exception as error:
        failed += 1
        print(f"Error processing record {record_number}: {error}")

record.close()

if not all_embeddings:
    raise RuntimeError("No embeddings were generated.")

all_embeddings = np.stack(all_embeddings).astype(np.float32)
all_labels = np.asarray(all_labels, dtype=np.int64)
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_EMBEDDINGS, all_embeddings)
np.save(OUTPUT_LABELS, all_labels)

print("Successful images:", len(all_embeddings))
print("Failed images:", failed)
print("Embedding shape:", all_embeddings.shape)
print("Labels shape:", all_labels.shape)
print("Embeddings saved to:", OUTPUT_EMBEDDINGS)
print("Labels saved to:", OUTPUT_LABELS)
