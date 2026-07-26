#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Iterable

BLOCK0_OVERRIDES = (
    "linear_attn_qkv_mixed",
    "z",
    "beta",
    "alpha",
    "linear_attn_out",
    "ffn_gate",
    "ffn_up",
    "ffn_out",
)

BLOCK3_OVERRIDES = (
    "Qcur_full",
    "Kcur_projected",
    "Vcur_projected",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_out",
)

REMAINING_MATVECS = (
    ("block0_ssm_out", 0, "blk.0.ssm_out.weight", "attn_out_norm", "linear_attn_out"),
    ("block0_ffn_gate", 0, "blk.0.ffn_gate.weight", "attn_post_norm", "ffn_gate"),
    ("block0_ffn_up", 0, "blk.0.ffn_up.weight", "attn_post_norm", "ffn_up"),
    ("block0_ffn_down", 0, "blk.0.ffn_down.weight", "ffn_swiglu", "ffn_out"),
    ("block3_attn_output", 3, "blk.3.attn_output.weight", "attn_gated", "attn_output"),
    ("block3_ffn_gate", 3, "blk.3.ffn_gate.weight", "attn_post_norm", "ffn_gate"),
    ("block3_ffn_up", 3, "blk.3.ffn_up.weight", "attn_post_norm", "ffn_up"),
    ("block3_ffn_down", 3, "blk.3.ffn_down.weight", "ffn_swiglu", "ffn_out"),
)


def canonical_f32(values: Iterable[object], context: str) -> list[float]:
    result: list[float] = []
    for index, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{context}[{index}] is non-finite")
        result.append(struct.unpack("<f", struct.pack("<f", value))[0])
    return result


def load_trace(path: Path, block_index: int, kind: str) -> dict[str, list[float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("block_index", -1)) != block_index:
        raise ValueError(f"{path}: expected block {block_index}")
    if document.get("kind") != kind:
        raise ValueError(f"{path}: expected kind {kind!r}")

    tensors: dict[str, list[float]] = {}
    for tensor in document.get("tensors", []):
        name = str(tensor["name"])
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        values = canonical_f32(tensor["values"], f"{path}:{name}")
        if int(tensor.get("count", len(values))) != len(values):
            raise ValueError(f"{path}: count mismatch for {name}")
        tensors[name] = values
    return tensors


def require_tensor(tensors: dict[str, list[float]], name: str, context: str) -> list[float]:
    try:
        return tensors[name]
    except KeyError as error:
        raise KeyError(f"{context}: missing tensor {name}") from error


def write_text_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(format(value, ".9g"))
            stream.write("\n")


def write_binary_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack(f"<{len(values)}f", *values))


def tensor_inventory(path: Path) -> dict[str, dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not document.get("verified"):
        raise ValueError(f"{path}: model inventory is not verified")
    tensors = {str(tensor["name"]): tensor for tensor in document["tensors"]}
    if len(tensors) != len(document["tensors"]):
        raise ValueError(f"{path}: duplicate tensor names")
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("block0_reference", type=Path)
    parser.add_argument("block3_reference", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    block0 = load_trace(args.block0_reference, 0, "recurrent")
    block3 = load_trace(args.block3_reference, 3, "attention")
    inventory = tensor_inventory(args.inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "block0_reference": str(args.block0_reference.resolve()),
        "block3_reference": str(args.block3_reference.resolve()),
        "overrides": {"block0": {}, "block3": {}},
    }

    for block_name, tensors, names in (
        ("block0", block0, BLOCK0_OVERRIDES),
        ("block3", block3, BLOCK3_OVERRIDES),
    ):
        for name in names:
            values = require_tensor(tensors, name, block_name)
            destination = args.output_dir / "overrides" / block_name / f"{name}.f32.txt"
            write_text_f32(destination, values)
            manifest["overrides"][block_name][name] = {
                "path": str(destination.resolve()),
                "count": len(values),
            }

    block3_input = args.output_dir / "block3-input.f32.txt"
    write_text_f32(block3_input, require_tensor(block3, "input", "block3"))
    manifest["block3_input"] = str(block3_input.resolve())

    job_dir = args.output_dir / "remaining-matvec"
    jobs_path = job_dir / "jobs.tsv"
    job_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "weight_name",
        "absolute_offset",
        "rows",
        "cols",
        "type_name",
        "input_file",
        "oracle_file",
        "reference_file",
        "output_prefix",
        "row_bytes",
    ]

    with jobs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for label, block_index, weight_name, input_name, target_name in REMAINING_MATVECS:
            trace = block0 if block_index == 0 else block3
            if weight_name not in inventory:
                raise KeyError(f"inventory missing {weight_name}")
            info = inventory[weight_name]
            dimensions = [int(value) for value in info["dimensions"]]
            if len(dimensions) != 2:
                raise ValueError(f"{weight_name}: expected rank 2")
            cols, rows = dimensions
            type_name = str(info["type_name"])
            if type_name not in {"q5_K", "q6_K"}:
                raise ValueError(f"{weight_name}: unsupported type {type_name}")
            byte_size = int(info["byte_size"])
            if byte_size % rows != 0:
                raise ValueError(f"{weight_name}: byte size is not row-aligned")
            row_bytes = byte_size // rows

            input_values = require_tensor(trace, input_name, label)
            target_values = require_tensor(trace, target_name, label)
            if len(input_values) != cols:
                raise ValueError(
                    f"{label}: input {input_name} has {len(input_values)} values, expected {cols}"
                )
            if len(target_values) != rows:
                raise ValueError(
                    f"{label}: target {target_name} has {len(target_values)} values, expected {rows}"
                )

            input_path = job_dir / f"{label}.input.f32"
            target_path = job_dir / f"{label}.reference.f32"
            output_prefix = job_dir / label
            write_binary_f32(input_path, input_values)
            write_binary_f32(target_path, target_values)

            # The probe requires both an Oracle and reference file. For these
            # later production-contract jobs, the reference is intentionally
            # supplied to both slots. Only q8k_production_vs_reference is used
            # by the dedicated summarizer.
            writer.writerow(
                {
                    "label": label,
                    "weight_name": weight_name,
                    "absolute_offset": int(info["absolute_offset"]),
                    "rows": rows,
                    "cols": cols,
                    "type_name": type_name,
                    "input_file": str(input_path.resolve()),
                    "oracle_file": str(target_path.resolve()),
                    "reference_file": str(target_path.resolve()),
                    "output_prefix": str(output_prefix.resolve()),
                    "row_bytes": row_bytes,
                }
            )

    manifest["remaining_matvec_jobs"] = str(jobs_path.resolve())
    manifest_path = args.output_dir / "bridge-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(manifest_path)
    print(jobs_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
