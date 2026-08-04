#!/usr/bin/env python3
"""Compare staged Qwen3.5 full-forward captures without conflating contracts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle_trace", type=Path)
    parser.add_argument("reference_trace", type=Path)
    parser.add_argument("--oracle-logits", type=Path)
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--mode", choices=("same-contract", "override-bridge"),
                        default="same-contract")
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def read_f32(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = float(stripped)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid float") from error
            if not math.isfinite(value):
                raise ValueError(f"{path}:{line_number}: non-finite float")
            values.append(value)
    if not values:
        raise ValueError(f"{path}: no floating-point values")
    return values


def vector_values(node: Any, context: str) -> list[float]:
    if not isinstance(node, dict) or not isinstance(node.get("values"), list):
        raise ValueError(f"{context}: missing values array")
    values = [float(value) for value in node["values"]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{context}: non-finite value")
    if node.get("count") != len(values):
        raise ValueError(f"{context}: count does not match values")
    return values


def contract_tuple(trace: dict[str, Any]) -> tuple[str, str]:
    contracts = trace.get("contracts")
    if not isinstance(contracts, dict):
        raise ValueError("capture has no contracts object")
    return (str(contracts.get("execution_projection", "")),
            str(contracts.get("execution_attention_cache", "")))


def verify_contracts(oracle: dict[str, Any], reference: dict[str, Any], mode: str) -> None:
    oracle_contract = contract_tuple(oracle)
    reference_contract = contract_tuple(reference)
    if mode == "same-contract":
        if oracle_contract != reference_contract:
            raise RuntimeError(
                "same-contract comparison refused: "
                f"Oracle={oracle_contract}, reference={reference_contract}")
        return

    contracts = oracle["contracts"]
    source_contract = (str(contracts.get("override_projection_source", "")),
                       str(contracts.get("override_attention_cache_source", "")))
    if oracle_contract[0] != "diagnostic-external-overrides":
        raise RuntimeError(
            "override-bridge comparison requires diagnostic-external-overrides execution")
    if source_contract != reference_contract:
        raise RuntimeError(
            "override source contracts do not match the reference: "
            f"source={source_contract}, reference={reference_contract}")


def compare_vector(name: str,
                   left: Sequence[float],
                   right: Sequence[float],
                   atol: float,
                   rtol: float) -> bool:
    if len(left) != len(right):
        print(f"{name}: COUNT_MISMATCH oracle={len(left)} reference={len(right)}")
        return False

    max_abs = 0.0
    max_rel = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    first_mismatch: int | None = None
    for index, (actual, expected) in enumerate(zip(left, right, strict=True)):
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1.0e-30)
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        squared_error += absolute * absolute
        squared_reference += expected * expected
        if first_mismatch is None and absolute > atol + rtol * abs(expected):
            first_mismatch = index

    relative_l2 = math.sqrt(squared_error) / max(math.sqrt(squared_reference), 1.0e-30)
    status = "MATCH" if first_mismatch is None else "MISMATCH"
    first = "none" if first_mismatch is None else str(first_mismatch)
    print(f"{name}: {status} count={len(left)} max_abs={max_abs:.9g} "
          f"max_rel={max_rel:.9g} rel_l2={relative_l2:.9g} first={first}")
    return first_mismatch is None


def block_map(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    blocks = trace.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("capture has no blocks array")
    result: dict[int, dict[str, Any]] = {}
    for node in blocks:
        if not isinstance(node, dict):
            raise ValueError("block capture must be an object")
        index = int(node["block_index"])
        if index in result:
            raise ValueError(f"duplicate block capture {index}")
        result[index] = node
    return result


def logits_from(trace: dict[str, Any], path: Path | None, context: str) -> list[float]:
    if path is not None:
        return read_f32(path)
    logits = trace.get("logits")
    if isinstance(logits, dict) and isinstance(logits.get("values"), list):
        return vector_values(logits, context)
    raise ValueError(f"{context}: provide a logits file or a full JSON capture")


def top_ids(values: Sequence[float], count: int = 20) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def main() -> int:
    arguments = parse_args()
    if arguments.atol < 0.0 or arguments.rtol < 0.0:
        raise ValueError("tolerances must be non-negative")

    oracle = load_json(arguments.oracle_trace)
    reference = load_json(arguments.reference_trace)
    try:
        verify_contracts(oracle, reference, arguments.mode)
    except RuntimeError as error:
        print(f"VERDICT=CONTRACT_MISMATCH\n{error}")
        return 2

    all_match = True
    all_match &= compare_vector(
        "embedding",
        vector_values(oracle.get("embedding"), "Oracle embedding"),
        vector_values(reference.get("embedding"), "reference embedding"),
        arguments.atol,
        arguments.rtol,
    )

    oracle_blocks = block_map(oracle)
    reference_blocks = block_map(reference)
    if oracle_blocks.keys() != reference_blocks.keys():
        print("blocks: INDEX_SET_MISMATCH")
        all_match = False
    for index in sorted(oracle_blocks.keys() & reference_blocks.keys()):
        oracle_node = oracle_blocks[index]
        reference_node = reference_blocks[index]
        if oracle_node.get("kind") != reference_node.get("kind"):
            print(f"block.{index}: KIND_MISMATCH")
            all_match = False
        all_match &= compare_vector(
            f"block.{index}.output",
            vector_values(oracle_node.get("output"), f"Oracle block {index}"),
            vector_values(reference_node.get("output"), f"reference block {index}"),
            arguments.atol,
            arguments.rtol,
        )

    all_match &= compare_vector(
        "final_norm",
        vector_values(oracle.get("final_norm"), "Oracle final norm"),
        vector_values(reference.get("final_norm"), "reference final norm"),
        arguments.atol,
        arguments.rtol,
    )

    oracle_logits = logits_from(oracle, arguments.oracle_logits, "Oracle logits")
    reference_logits = logits_from(reference, arguments.reference_logits, "reference logits")
    logits_match = compare_vector(
        "logits", oracle_logits, reference_logits, arguments.atol, arguments.rtol)
    all_match &= logits_match

    if len(oracle_logits) == len(reference_logits):
        oracle_argmax = max(range(len(oracle_logits)), key=oracle_logits.__getitem__)
        reference_argmax = max(range(len(reference_logits)), key=reference_logits.__getitem__)
        oracle_top = top_ids(oracle_logits)
        reference_top = top_ids(reference_logits)
        print(f"argmax: oracle={oracle_argmax} reference={reference_argmax} "
              f"match={str(oracle_argmax == reference_argmax).lower()}")
        denominator = min(20, len(oracle_logits), len(reference_logits))
        print(f"top20_id_overlap={len(set(oracle_top) & set(reference_top))}/{denominator}")

    print("VERDICT=PHASE2D_FORWARD_MATCH" if all_match
          else "VERDICT=PHASE2D_FORWARD_MISMATCH")
    return 0 if all_match else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"compare_qwen35_forward_captures.py: {error}", file=sys.stderr)
        raise SystemExit(2) from error
