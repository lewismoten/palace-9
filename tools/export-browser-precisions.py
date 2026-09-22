#!/usr/bin/env python3
"""Export Palace-9 deployment tensors as provenance-preserving JSON envelopes.

The JSON stores original tensor byte payloads (base64), never expanded decimal arrays.
Browser code decodes each payload using the declared GGML storage type.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/palace-9-local-v1"
OUT = RELEASE / "browser"
ALIGN = 32

# GGUF v3 value type tags.
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_NAMES = {0: "F32", 1: "F16", 12: "Q4_K", 14: "Q6_K"}
TYPE_BYTES = {0: lambda n: n * 4, 1: lambda n: n * 2, 12: lambda n: n // 256 * 144, 14: lambda n: n // 256 * 210}

SIMPLE = {
    "token_embd.weight": ("model.embed_tokens.weight", True),
    "blk.0.attn_norm.weight": ("model.layers.0.input_layernorm.weight", False),
    "blk.0.attn_q.weight": ("model.layers.0.self_attn.q_proj.weight", True),
    "blk.0.attn_q.bias": ("model.layers.0.self_attn.q_proj.bias", False),
    "blk.0.attn_k.weight": ("model.layers.0.self_attn.k_proj.weight", True),
    "blk.0.attn_k.bias": ("model.layers.0.self_attn.k_proj.bias", False),
    "blk.0.attn_v.weight": ("model.layers.0.self_attn.v_proj.weight", True),
    "blk.0.attn_v.bias": ("model.layers.0.self_attn.v_proj.bias", False),
    "blk.0.attn_output.weight": ("model.layers.0.self_attn.o_proj.weight", True),
    "blk.0.ffn_norm.weight": ("model.layers.0.post_attention_layernorm.weight", False),
    "blk.0.ffn_gate_inp.weight": ("model.layers.0.mlp.gate.weight", True),
    "blk.0.ffn_gate_inp_shexp.weight": ("model.layers.0.mlp.shared_expert_gate.weight", False),
    "blk.0.output_norm.weight": ("model.norm.weight", False),
    "output_norm.weight": ("model.norm.weight", False),
    "output.weight": ("lm_head.weight", True),
}
EXPERTS = {
    "blk.0.ffn_gate_exps.weight": ("gate_proj", [18, 36]),
    "blk.0.ffn_up_exps.weight": ("up_proj", [18, 36]),
    "blk.0.ffn_down_exps.weight": ("down_proj", [36, 18]),
}
SHARED = {
    "blk.0.ffn_gate_shexp.weight": ("gate_proj", [5632, 36]),
    "blk.0.ffn_up_shexp.weight": ("up_proj", [5632, 36]),
    "blk.0.ffn_down_shexp.weight": ("down_proj", [36, 5632]),
}

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def u32(data, at): return struct.unpack_from("<I", data, at)[0]
def u64(data, at): return struct.unpack_from("<Q", data, at)[0]

def string(data, at):
    length = u64(data, at); at += 8
    return data[at:at + length].decode("utf-8"), at + length

def skip_value(data, at, typ):
    if typ == 8:
        _, at = string(data, at)
        return at
    if typ == 9:
        subtype = u32(data, at); count = u64(data, at + 4); at += 12
        for _ in range(count): at = skip_value(data, at, subtype)
        return at
    return at + FIXED[typ]

def parse_gguf(path: Path):
    data = path.read_bytes()
    if data[:4] != b"GGUF": raise ValueError(f"not GGUF: {path}")
    if u32(data, 4) != 3: raise ValueError("only GGUF v3 is supported")
    tensor_count, kv_count = u64(data, 8), u64(data, 16)
    at, alignment = 24, ALIGN
    for _ in range(kv_count):
        key, at = string(data, at); typ = u32(data, at); at += 4
        if key == "general.alignment" and typ == 4: alignment = u32(data, at)
        at = skip_value(data, at, typ)
    directory = []
    for _ in range(tensor_count):
        name, at = string(data, at); dims = u32(data, at); at += 4
        shape = [u64(data, at + 8 * i) for i in range(dims)]; at += 8 * dims
        typ, offset = u32(data, at), u64(data, at + 4); at += 12
        directory.append((name, shape, typ, offset))
    data_start = (at + alignment - 1) // alignment * alignment
    tensors = {}
    for name, shape, typ, offset in directory:
        count = 1
        for dim in shape: count *= dim
        if typ not in TYPE_BYTES: raise ValueError(f"unsupported GGML type {typ} for {name}")
        size = TYPE_BYTES[typ](count)
        payload = data[data_start + offset:data_start + offset + size]
        if len(payload) != size: raise ValueError(f"truncated {name}")
        tensors[name] = {"shape": shape, "type": TYPE_NAMES[typ], "bytes": payload}
    return tensors

def parse_safetensors(path: Path):
    raw = path.read_bytes(); header_len = u64(raw, 0)
    header = json.loads(raw[8:8 + header_len]); start = 8 + header_len
    result = {}
    for name, entry in header.items():
        if name == "__metadata__": continue
        if entry["dtype"] != "F32": raise ValueError(f"expected F32 source tensor: {name}")
        begin, end = entry["data_offsets"]
        result[name] = {"shape": entry["shape"], "type": "F32", "bytes": raw[start + begin:start + end]}
    return result

def add_source(sources, name, tensor):
    sources[name] = {
        "type": tensor["type"], "gguf_shape": tensor["shape"],
        "sha256": sha(tensor["bytes"]), "bytes": base64.b64encode(tensor["bytes"]).decode("ascii"),
    }

def gguf_aliases(tensors):
    aliases = {}
    for source, (target, reverse_shape) in SIMPLE.items():
        if source not in tensors: continue
        shape = list(reversed(tensors[source]["shape"])) if reverse_shape else tensors[source]["shape"]
        aliases[target] = {"source": source, "shape": shape}
    for source, (part, shape) in EXPERTS.items():
        tensor = tensors[source]; width = shape[0] * shape[1]
        for expert in range(9):
            aliases[f"model.layers.0.mlp.experts.{expert}.{part}.weight"] = {
                "source": source, "shape": shape, "offset_elements": expert * width,
            }
    for source, (part, shape) in SHARED.items():
        aliases[f"model.layers.0.mlp.shared_expert.{part}.weight"] = {"source": source, "shape": shape}
    return aliases

def write_manifest(name, source_path: Path, tensors, aliases, artifact_kind):
    source_bytes = source_path.read_bytes()
    manifest = {
        "format": "palace9-packed-tensors-v1", "precision": name,
        "claim": "exact packed deployment artifact decoded locally in the browser" if artifact_kind == "gguf" else "exact FP32 source checkpoint bytes decoded locally in the browser",
        "source": {"path": source_path.name, "sha256": sha(source_bytes), "size_bytes": len(source_bytes), "kind": artifact_kind},
        "sources": {}, "tensors": aliases,
    }
    for tensor_name, tensor in tensors.items(): add_source(manifest["sources"], tensor_name, tensor)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")

def main():
    fp32_path = RELEASE / "source-checkpoint/model.safetensors"
    fp32 = parse_safetensors(fp32_path)
    write_manifest("fp32", fp32_path, fp32, {name: {"source": name, "shape": tensor["shape"]} for name, tensor in fp32.items()}, "safetensors")
    for name, filename in [("f16", "palace9-qwen2moe-raw-history-f16.gguf"), ("q6_k", "palace9-qwen2moe-raw-history-q6_k.gguf"), ("q4_k_m", "palace9-qwen2moe-raw-history-q4_k_m.gguf")]:
        path = RELEASE / filename; tensors = parse_gguf(path)
        write_manifest(name, path, tensors, gguf_aliases(tensors), "gguf")

if __name__ == "__main__": main()
