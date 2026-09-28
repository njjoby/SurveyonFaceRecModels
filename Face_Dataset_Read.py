import mxnet as mx

rec_path = "CASIA-WebFace/train.rec"
idx_path = "CASIA-WebFace/train.idx"

record = mx.recordio.MXIndexedRecordIO(
    idx_path,
    rec_path,
    "r"
)

print("Number of records:", len(record.keys))

item = record.read_idx(0)

print("First record type:", type(item))
print("First record size:", len(item))