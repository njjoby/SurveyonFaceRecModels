"""Generate face embeddings with the pretrained TransFace-S model.

Downloads the authors' MS1MV2 checkpoint on first use. The script performs
inference only and does not train a model.
"""

import gc
import os

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
OUTPUT_DIR = "Embeddings Generated TransFace/CASIA-WebFace-400K"
OUTPUT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "transface_embeddings.npy")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "transface_labels.npy")

MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "ms1mv2_model_TransFace_S.pt")
GOOGLE_DRIVE_FILE_ID = "1UZWCg7jNESDv8EWs7mxQSswCMGbAZNF4"
CHECKPOINT_DOWNLOAD_PAGE = (
    "https://drive.google.com/file/d/"
    + GOOGLE_DRIVE_FILE_ID
    + "/view"
)

INPUT_SIZE = 112
EMBEDDING_SIZE = 512
PATCH_SIZE = 9
NUM_PATCHES = (INPUT_SIZE // PATCH_SIZE) ** 2


# ============================================================
# TransFace-S backbone
# Architecture follows the official TransFace implementation.
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        # Stochastic depth is inactive at inference and has no learned weights.
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=112, patch_size=9, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"Expected {self.img_size} input, got {x.shape[-2:]}")
        return self.proj(x).flatten(2).transpose(1, 2)


class TransFaceS(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 512
        self.num_classes = EMBEDDING_SIZE
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(INPUT_SIZE, PATCH_SIZE, 3, embed_dim)
        self.mask_ratio = 0.0
        self.using_checkpoint = False
        self.num_patches = NUM_PATCHES
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=8, drop_path=0.05)
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.feature = nn.Sequential(
            nn.Linear(embed_dim * NUM_PATCHES, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
            nn.Linear(embed_dim, EMBEDDING_SIZE, bias=False),
            nn.BatchNorm1d(EMBEDDING_SIZE, eps=2e-5),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.senet = nn.Sequential(
            nn.Linear(embed_dim * NUM_PATCHES, NUM_PATCHES, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(NUM_PATCHES, NUM_PATCHES, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.float())

        batch = x.shape[0]
        flat = x.reshape(batch, NUM_PATCHES * self.embed_dim)
        patch_weights = self.senet(flat)
        attention_weights = patch_weights.softmax(dim=1)
        weighted_patches = x * patch_weights.reshape(batch, NUM_PATCHES, 1)
        features = self.feature(weighted_patches.reshape(batch, -1))
        patch_entropy = torch.std(x, dim=2)
        return features, attention_weights, patch_entropy


def download_checkpoint():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 0:
        return

    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Install gdown to download the authors' pretrained TransFace weights:\n"
            ".venv/bin/pip install gdown"
        ) from error

    print("Downloading pretrained TransFace-S checkpoint...")
    try:
        result = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=CHECKPOINT_PATH,
            quiet=False,
        )
    except Exception as error:
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        raise RuntimeError(
            "Could not download the pretrained TransFace-S checkpoint. "
            "Check internet/DNS access to drive.google.com, or manually download "
            f"the checkpoint from {CHECKPOINT_DOWNLOAD_PAGE} and save it as "
            f"{CHECKPOINT_PATH}."
        ) from error

    if not result or not os.path.isfile(CHECKPOINT_PATH) or os.path.getsize(CHECKPOINT_PATH) == 0:
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        raise RuntimeError(
            "TransFace checkpoint download failed. Manually download the "
            f"TransFace-S checkpoint from {CHECKPOINT_DOWNLOAD_PAGE} and save it as "
            + CHECKPOINT_PATH
        )


def load_model(device):
    download_checkpoint()
    model = TransFaceS()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unrecognized TransFace checkpoint format")

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
            "Downloaded checkpoint does not match TransFace-S. "
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
            packed = header = image_bytes = image_array = image = image_tensor = output = None
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
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

                # Matches the official inference preprocessing: RGB, range [-1, 1].
                image = image.astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image_tensor = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                ).unsqueeze(0).to(device)

                with torch.inference_mode():
                    output = model(image_tensor)
                    embedding = output[0].squeeze(0).cpu().numpy()
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding.size != EMBEDDING_SIZE or not np.isfinite(embedding).all():
                    raise ValueError("Model returned an invalid TransFace embedding")
                norm = np.linalg.norm(embedding)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("Model returned a zero or invalid embedding norm")

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
                del packed, header, image_bytes, image_array, image, image_tensor, output
                if record_number % 500 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        if successful == 0:
            raise RuntimeError("No TransFace embeddings were generated.")

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

    print("\nTransFace embedding extraction completed")
    print("Successful images:", successful)
    print("Failed images:", failed)
    print("Embeddings shape:", (successful, EMBEDDING_SIZE))
    print("Labels shape:", (successful,))
    print("Embeddings saved to:", OUTPUT_EMBEDDINGS)
    print("Labels saved to:", OUTPUT_LABELS)


if __name__ == "__main__":
    main()
