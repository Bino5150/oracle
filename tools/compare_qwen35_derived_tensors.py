#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

INJECTED = {
    "recurrent": (
        "linear_attn_qkv_mixed",
        "z",
        "beta",
        "alpha",
        "linear_attn_out",
        "ffn_gate",
        "ffn_up",
        "ffn_out",
    ),
    "attention": (
        "Qcur_full",
        "Kcur_projected",
        "Vcur_projected",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_out",
    ),
}

DERIVED = {
    "recurrent": (
        "input",
        "attn_norm",
        "beta_sigmoid",
        "a_softplus",
        "gate",
        "conv_output_raw",
        "conv_output_silu",
        "q_conv",
        "k_conv",
        "v_conv",
        "q_conv_predelta",
        "k_conv_predelta",
        "v_conv_predelta",
        "state_predelta",
        "recurrent_output",
        "state_postdelta",
        "attn_out_norm",
        "attn_residual",
        "attn_post_norm",
        "ffn_swiglu",
        "post_ffn",
    ),
    "attention": (
        "input",
        "attn_norm",
        "Qcur_reshaped",
        "gate_reshaped",
        "Qcur_normed",
        "Kcur_normed",
        "Qcur",
        "Kcur",
        "Vcur",
        "attn_pregate",
        "gate_sigmoid",
        "attn_gated",
        "attn_residual",
        "attn_post_norm",
        "ffn_swiglu",
        "post_ffn",
    ),
}


def load(path: Path) -> tuple[dict[str, object], dict[str, list[float]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    tensors: dict[str, list[float]] = {}
    for tensor in document["tensors"]:
        name = str(tensor["name"])
        values = [float(value) for value in tensor["values"]]
        if len(values) != int(tensor.get("count", len(values))):
            raise ValueError(f"{path}: count mismatch for {name}")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{path}: non-finite value in {name}")
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        tensors[name] = values
    return document, tensors


def compare(actual: list[float], expected: list[float], atol: float, rtol: float):
    if len(actual) != len(expected):
        return math.inf, math.inf, 0
    max_abs = 0.0
    max_rel = 0.0
    first = None
    for index, (left, right) in enumerate(zip(actual, expected)):
        error = abs(left - right)
        relative = error / max(abs(right), 1.0e-30)
        max_abs = max(max_abs, error)
        max_rel = max(max_rel, relative)
        if first is None and error > atol + rtol * abs(right):
            first = index
    return max_abs, max_rel, first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    args = parser.parse_args()

    oracle_doc, oracle = load(args.oracle)
    reference_doc, reference = load(args.reference)
    kind = str(oracle_doc.get("kind"))
    if kind != reference_doc.get("kind") or kind not in INJECTED:
        raise ValueError("trace kind mismatch or unsupported kind")
    if oracle_doc.get("block_index") != reference_doc.get("block_index"):
        raise ValueError("block index mismatch")
    if oracle_doc.get("position") != reference_doc.get("position"):
        raise ValueError("position mismatch")

    lines = ["mode\ttensor\tcount\tmax_abs\tmax_rel\tfirst_mismatch"]
    failures = 0
    for mode, names in (("injected", INJECTED[kind]), ("derived", DERIVED[kind])):
        for name in names:
            if name not in oracle or name not in reference:
                lines.append(f"{mode}\t{name}\t0\tinf\tinf\tmissing")
                failures += 1
                continue
            maximum, relative, first = compare(
                oracle[name], reference[name], args.atol, args.rtol
            )
            mismatch = "none" if first is None else str(first)
            lines.append(
                f"{mode}\t{name}\t{len(oracle[name])}\t"
                f"{maximum:.12g}\t{relative:.12g}\t{mismatch}"
            )
            if first is not None:
                failures += 1

    lines.append("")
    if failures == 0:
        lines.append("VERDICT=CONTRACT_BRIDGE_MATCH")
        lines.append(
            "All injected matvec outputs and all derived post-projection tensors "
            f"match within atol={args.atol:g}, rtol={args.rtol:g}."
        )
    else:
        lines.append("VERDICT=CONTRACT_BRIDGE_MISMATCH")
        lines.append(f"failure_count={failures}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
