#!/usr/bin/env python3
"""Compare two Oracle-compatible Qwen3.5 block trace JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("tensors"), list):
        raise ValueError(f"{path}: trace must contain a tensors array")
    return value


def tensor_map(trace: dict[str, Any], path: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for item in trace["tensors"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"{path}: malformed tensor record")
        name = item["name"]
        values = item.get("values")
        if not isinstance(values, list) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            raise ValueError(f"{path}: tensor {name!r} has invalid values")
        if name in result:
            raise ValueError(f"{path}: duplicate tensor name {name!r}")
        result[name] = [float(value) for value in values]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--tensor", action="append", default=[])
    args = parser.parse_args()

    oracle_trace = load_trace(args.oracle)
    reference_trace = load_trace(args.reference)
    for field in ("block_index", "kind", "position"):
        if field in oracle_trace and field in reference_trace and oracle_trace[field] != reference_trace[field]:
            raise ValueError(
                f"trace {field} mismatch: {oracle_trace[field]!r} != {reference_trace[field]!r}"
            )

    oracle_tensors = tensor_map(oracle_trace, args.oracle)
    reference_tensors = tensor_map(reference_trace, args.reference)

    names = args.tensor or sorted(set(oracle_tensors) & set(reference_tensors))
    if not names:
        raise ValueError("no common tensors to compare")

    failed = False
    print("tensor\tcount\tmax_abs\tmax_rel\tfirst_mismatch")
    for name in names:
        if name not in oracle_tensors or name not in reference_tensors:
            print(f"{name}\tmissing")
            failed = True
            continue
        left = oracle_tensors[name]
        right = reference_tensors[name]
        if len(left) != len(right):
            print(f"{name}\tlength {len(left)} != {len(right)}")
            failed = True
            continue

        max_abs = 0.0
        max_rel = 0.0
        first_mismatch: int | None = None
        for index, (actual, expected) in enumerate(zip(left, right)):
            if not math.isfinite(actual) or not math.isfinite(expected):
                equal = actual == expected or (math.isnan(actual) and math.isnan(expected))
                if not equal and first_mismatch is None:
                    first_mismatch = index
                if not equal:
                    failed = True
                continue
            absolute = abs(actual - expected)
            relative = absolute / max(abs(expected), 1.0e-30)
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            if absolute > args.atol + args.rtol * abs(expected) and first_mismatch is None:
                first_mismatch = index
                failed = True

        mismatch_text = "none" if first_mismatch is None else str(first_mismatch)
        print(f"{name}\t{len(left)}\t{max_abs:.9g}\t{max_rel:.9g}\t{mismatch_text}")

    if not args.tensor:
        missing_oracle = sorted(set(reference_tensors) - set(oracle_tensors))
        missing_reference = sorted(set(oracle_tensors) - set(reference_tensors))
        if missing_oracle:
            print("missing from Oracle: " + ", ".join(missing_oracle))
            failed = True
        if missing_reference:
            print("missing from reference: " + ", ".join(missing_reference))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
