"""Layer-addressable lossless BF16 Replica, not a second KV Artifact."""
from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path
import struct

MAGIC = b"PKVLAYER1\0"


def write_layer_replica(stream, layers):
    import torch
    if not layers or len(layers[0]) != 2:
        raise ValueError("canonical Replica requires at least one K/V layer")
    shape = tuple(layers[0][0].shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("invalid layer geometry")
    header = json.dumps({"shape": shape, "layers": len(layers), "dtype": "bfloat16",
                         "k_semantics": "pre_rope", "v_semantics": "raw"}, sort_keys=True).encode()
    stream.write(MAGIC + struct.pack("<I", len(header)) + header)
    for pair in layers:
        if len(pair) != 2:
            raise ValueError("missing canonical K/V")
        for tensor in pair:
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ValueError("noncanonical layer shape/dtype")
            stream.write(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


class LayerFile(Sequence):
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            if stream.read(len(MAGIC)) != MAGIC:
                raise ValueError("not a layer Replica")
            length = struct.unpack("<I", stream.read(4))[0]
            if not 0 < length <= 65536:
                raise ValueError("invalid layer index length")
            metadata = json.loads(stream.read(length))
            self.offset = stream.tell()
        self.shape = tuple(metadata["shape"])
        self.layer_count = metadata["layers"]
        if (metadata["dtype"] != "bfloat16" or metadata["k_semantics"] != "pre_rope"
                or metadata["v_semantics"] != "raw" or len(self.shape) != 3
                or any(type(x) is not int or x < 1 for x in self.shape)
                or type(self.layer_count) is not int or self.layer_count < 1):
            raise ValueError("invalid layer index geometry")
        self.tensor_bytes = math.prod(self.shape) * 2
        self.full_kv_bytes = self.tensor_bytes * 2 * self.layer_count
        if self.path.stat().st_size != self.offset + self.full_kv_bytes:
            raise RuntimeError("truncated/oversized KV Replica")

    @classmethod
    def recognizes(cls, path):
        with Path(path).open("rb") as stream:
            return stream.read(len(MAGIC)) == MAGIC

    def __len__(self):
        return self.layer_count

    def read_into(self, index, pair):
        import torch
        if not 0 <= index < len(self):
            raise IndexError(index)
        if len(pair) != 2:
            raise ValueError("expected K/V staging pair")
        with self.path.open("rb") as stream:
            stream.seek(self.offset + index * 2 * self.tensor_bytes)
            for tensor in pair:
                if (tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
                        or tuple(tensor.shape) != self.shape or not tensor.is_contiguous()):
                    raise ValueError("invalid CPU staging buffer")
                target = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
                n = 0
                while n < len(target):
                    count = stream.readinto(target[n:])
                    if not count:
                        raise RuntimeError("short layer read")
                    n += count
        return pair

    def __getitem__(self, index):
        import torch
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pair = tuple(torch.empty(self.shape, dtype=torch.bfloat16) for _ in range(2))
        return self.read_into(index, pair)
