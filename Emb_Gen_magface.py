"""Generate MagFace embeddings from an MXNet RecordIO face dataset.

Uses the pretrained MagFace iResNet-18 checkpoint trained on CASIA-WebFace.
The checkpoint is downloaded on first use; this script does not train a model.
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
OUTPUT_DIR = "Embeddings Generated MagFace/CASIA-WebFace-400K"
OUTPUT_EMBEDDINGS = os.path.join(OUTPUT_DIR, "magface_embeddings.npy")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "magface_labels.npy")

MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "magface_iresnet18_casia_dp.pth")
# Official MagFace Model Zoo checkpoint (iResNet-18, trained on CASIA-WebFace).
GOOGLE_DRIVE_FILE_ID = "18pSIQOHRBQ-srrYfej20S5M8X8b_7zb9"
EMBEDDING_SIZE = 512
INPUT_SIZE = (112, 112)


# ============================================================
# MagFace iResNet-18 feature network
# Architecture follows the official MagFace repository.
# ============================================================

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=2e-5, momentum=0.9)
        self.conv1 = conv3x3(in_channels, channels)
        self.bn2 = nn.BatchNorm2d(channels, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn3 = nn.BatchNorm2d(channels, eps=2e-5, momentum=0.9)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class MagFaceIResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, 2, stride=2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, EMBEDDING_SIZE)
        self.features = nn.BatchNorm1d(EMBEDDING_SIZE, eps=2e-5, momentum=0.9)

    def _make_layer(self, channels, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, channels, stride),
                nn.BatchNorm2d(channels, eps=2e-5, momentum=0.9),
            )
        layers = [IBasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        return self.features(self.fc(x))


def download_checkpoint():
    """Fetch the official model checkpoint from Google Drive if absent."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(CHECKPOINT_PATH):
        return

    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "The MagFace checkpoint downloader is missing. Install it with:\n"
            ".venv/bin/pip install gdown"
        ) from error

    print("Downloading pretrained MagFace iResNet-18 checkpoint...")
    result = gdown.download(
        id=GOOGLE_DRIVE_FILE_ID,
        output=CHECKPOINT_PATH,
        quiet=False,
    )
    if not result or not os.path.isfile(CHECKPOINT_PATH) or os.path.getsize(CHECKPOINT_PATH) == 0:
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        raise RuntimeError(
            "MagFace checkpoint download failed. You can download it from the "
            "official MagFace Model Zoo and place it at: " + CHECKPOINT_PATH
        )


def load_pretrained_model(device):
    download_checkpoint()
    model = MagFaceIResNet18()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()

    # Official checkpoints may include DataParallel and wrapper prefixes.
    # Match each backbone tensor by its terminal parameter name and shape.
    loaded = {}
    for checkpoint_key, value in state_dict.items():
        normalized_key = checkpoint_key
        while normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module."):]

        for model_key, target in model_state.items():
            if (
                normalized_key == model_key
                or normalized_key.endswith("." + model_key)
            ) and tuple(value.shape) == tuple(target.shape):
                loaded[model_key] = value

    missing = set(model_state) - set(loaded)
    del checkpoint, state_dict
    if missing:
        raise RuntimeError(
            "The downloaded checkpoint does not match the MagFace iResNet-18 "
            f"architecture ({len(missing)} model tensors were not found)."
        )

    model.load_state_dict(loaded, strict=True)
    del loaded
    return model.to(device).eval()


def parse_label(raw_label, record_number):
    label_array = np.asarray(raw_label).reshape(-1)
    if label_array.size == 0:
        raise ValueError(f"Record {record_number} has no identity label")
    return int(label_array[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_pretrained_model(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    record = mx.recordio.MXIndexedRecordIO(IDX_PATH, REC_PATH, "r")
    number_of_records = len(record.keys)
    if number_of_records == 0:
        record.close()
        raise RuntimeError("The RecordIO dataset contains no records.")

    # Memory-map outputs rather than keeping the complete embedding set in RAM.
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
            packed = header = image_bytes = image_array = image = face_tensor = None
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

                # Official MagFace inference uses ToTensor(): BGR values in [0, 1].
                image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
                face_tensor = torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                ).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze_(0)

                with torch.inference_mode():
                    embedding = model(face_tensor).squeeze(0).cpu().numpy()
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding.size != EMBEDDING_SIZE or not np.isfinite(embedding).all():
                    raise ValueError("Model returned an invalid MagFace feature")

                # Preserve MagFace feature magnitude, which carries its quality signal.
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
                del packed, header, image_bytes, image_array, image, face_tensor, embedding
                if record_number % 500 == 0:
                    gc.collect()

        if successful == 0:
            raise RuntimeError("No MagFace embeddings were generated.")

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

    print("\nMagFace embedding extraction completed")
    print("Successful images:", successful)
    print("Failed images:", failed)
    print("Embeddings shape:", (successful, EMBEDDING_SIZE))
    print("Labels shape:", (successful,))
    print("Embeddings saved to:", OUTPUT_EMBEDDINGS)
    print("Labels saved to:", OUTPUT_LABELS)


if __name__ == "__main__":
    main()
