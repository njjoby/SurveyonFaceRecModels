"""Generate embeddings with Face Transformer (FaceT) for face recognition.

Downloads the authors' pretrained ViT-P8S8 CosFace checkpoint on first run.
This is inference only; no local model training is performed.
"""

import gc
import glob
import os
import re

import cv2
import mxnet as mx
import numpy as np
import torch
from torch import nn


# ============================================================
# Configuration
# ============================================================

REC_PATH = "CASIA-WebFace-400K/train.rec"
IDX_PATH = "CASIA-WebFace-400K/train.idx"
OUTPUT_DIR = "Embeddings Generated FaceT/CASIA-WebFace-400K"
OUTPUT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "facet_embeddings.npy")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "facet_labels.npy")

# Official Face-Transformer ViT-P8S8 pretrained model folder.
WEIGHTS_DIR = os.path.join("models", "FaceT_ViT-P8S8_ms1m_cosface")
WEIGHTS_FOLDER_ID = "1U7MDZSS38cMIvtEWohaLAH4j7JgBNy0T"
CHECKPOINT_PATH = ""  # Optional: set this to one specific downloaded .pth file.

INPUT_SIZE = 112
EMBEDDING_SIZE = 512
NUM_TRAIN_IDENTITIES = 93431  # Required to instantiate the training checkpoint head.


# ============================================================
# Face Transformer ViT-P8S8 architecture
# The CosFace classification head is instantiated for checkpoint compatibility;
# inference returns the 512-D embedding before that head.
# ============================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        batch, tokens, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.reshape(batch, tokens, self.heads, -1).permute(0, 2, 1, 3)
            for tensor in qkv
        ]
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, -1)
        return self.to_out(output)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads, dim_head, dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = attention(x)
            x = feed_forward(x)
        return x


class CosFaceHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x, label=None):
        return x


class FaceT(nn.Module):
    def __init__(self):
        super().__init__()
        patch_size = 8
        num_patches = (INPUT_SIZE // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, 512))
        self.patch_to_embedding = nn.Linear(patch_dim, 512)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.dropout = nn.Dropout(0.1)
        self.transformer = Transformer(
            dim=512,
            depth=20,
            heads=8,
            dim_head=64,
            mlp_dim=2048,
            dropout=0.1,
        )
        self.pool = "cls"
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(512))
        self.loss_type = "CosFace"
        self.GPU_ID = None
        self.loss = CosFaceHead(512, NUM_TRAIN_IDENTITIES)

    def forward(self, image, label=None):
        patch_size = self.patch_size
        batch, channels, height, width = image.shape
        x = image.reshape(
            batch,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, 3 * patch_size * patch_size)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embedding[:, : x.shape[1]])
        x = self.transformer(x)
        x = self.to_latent(x[:, 0])
        embedding = self.mlp_head(x)
        if label is not None:
            return self.loss(embedding, label), embedding
        return embedding


def download_checkpoint():
    global CHECKPOINT_PATH
    if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
        return CHECKPOINT_PATH

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    checkpoints = glob.glob(os.path.join(WEIGHTS_DIR, "**", "*.pth"), recursive=True)
    if not checkpoints:
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError(
                "Install gdown to download the authors' pretrained FaceT weights:\n"
                ".venv/bin/pip install gdown"
            ) from error

        print("Downloading pretrained FaceT ViT-P8S8 weights from the authors' Drive folder...")
        downloaded = gdown.download_folder(
            id=WEIGHTS_FOLDER_ID,
            output=WEIGHTS_DIR,
            quiet=False,
        )
        checkpoints = [
            path for path in (downloaded or [])
            if isinstance(path, str) and path.lower().endswith(".pth")
        ]
        if not checkpoints:
            checkpoints = glob.glob(os.path.join(WEIGHTS_DIR, "**", "*.pth"), recursive=True)

    if not checkpoints:
        raise FileNotFoundError(
            "No .pth checkpoint was found in the FaceT weights folder. "
            "Download the pretrained ViT-P8S8 checkpoint from the authors' "
            "Face-Transformer repository and set CHECKPOINT_PATH."
        )

    # Prefer the final checkpoint if the shared Drive folder contains several snapshots.
    def checkpoint_rank(path):
        numbers = [int(number) for number in re.findall(r"\d+", os.path.basename(path))]
        return (max(numbers, default=-1), os.path.basename(path))

    CHECKPOINT_PATH = max(checkpoints, key=checkpoint_rank)
    print("Using pretrained FaceT checkpoint:", CHECKPOINT_PATH)
    return CHECKPOINT_PATH


def load_model(device):
    checkpoint_path = download_checkpoint()
    model = FaceT()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unrecognized FaceT checkpoint format")

    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        clean_key = key
        while clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        if clean_key in model_state and tuple(value.shape) == tuple(model_state[clean_key].shape):
            compatible[clean_key] = value

    missing = [key for key in model_state if key not in compatible]
    del checkpoint, state_dict
    if missing:
        raise RuntimeError(
            "Downloaded checkpoint is not compatible with FaceT ViT-P8S8. "
            f"Missing {len(missing)} model tensors; first missing keys: {missing[:5]}"
        )

    model.load_state_dict(compatible, strict=True)
    del compatible
    return model.to(device).eval()


def parse_label(raw_label, record_number):
    label_array = np.asarray(raw_label).reshape(-1)
    if label_array.size == 0:
        raise ValueError(f"Record {record_number} has no identity label")
    return int(label_array[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    record = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, "r")
    number_of_records = len(record.keys)
    if number_of_records == 0:
        record.close()
        raise RuntimeError("The RecordIO dataset contains no records.")

    temporary_embeddings = OUTPUT_EMBEDDINGS + ".working.npy"
    temporary_labels = OUTPUT_LABELS + ".working.npy"
    embeddings_map = np.lib.format.open_memmap(
        temporary_embeddings, mode="w+", dtype=np.float32,
        shape=(number_of_records, EMBEDDING_SIZE),
    )
    labels_map = np.lib.format.open_memmap(
        temporary_labels, mode="w+", dtype=np.int64,
        shape=(number_of_records,),
    )

    successful = 0
    failed = 0
    try:
        print(f"Processing {number_of_records} RecordIO entries")
        for record_number, key in enumerate(record.keys, start=1):
            packed = header = image_bytes = image_array = image = image_tensor = embedding = None
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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

                # FaceT's released inference path uses raw RGB pixels in [0, 255].
                image_tensor = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                ).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    embedding = model(image_tensor).squeeze(0).cpu().numpy()
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding.size != EMBEDDING_SIZE or not np.isfinite(embedding).all():
                    raise ValueError("Model returned an invalid FaceT embedding")
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    raise ValueError("Model returned a zero embedding")

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
                del packed, header, image_bytes, image_array, image, image_tensor, embedding
                if record_number % 500 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        if successful == 0:
            raise RuntimeError("No FaceT embeddings were generated.")

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

    print("\nFaceT embedding extraction completed")
    print("Successful images:", successful)
    print("Failed images:", failed)
    print("Embeddings shape:", (successful, EMBEDDING_SIZE))
    print("Labels shape:", (successful,))
    print("Embeddings saved to:", OUTPUT_EMBEDDINGS)
    print("Labels saved to:", OUTPUT_LABELS)


if __name__ == "__main__":
    main()
