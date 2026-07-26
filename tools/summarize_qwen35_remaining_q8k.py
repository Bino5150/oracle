#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance", type=float, default=5.0e-5)
    args = parser.parse_args()

    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])

    lines = [
        "label\tq8k_production_vs_reference_max_abs\t"
        "q8k_production_vs_reference_rel_l2\t"
        "q8k_production_vs_generic_max_abs"
    ]
    failures = 0
    for label in labels:
        def choose(mode: str, target: str) -> dict[str, str]:
            for row in rows:
                if row["label"] == label and row["mode"] == mode and row["target"] == target:
                    return row
            raise KeyError((label, mode, target))

        production = choose("q8k_production", "reference")
        generic = choose("q8k_prod_vs_generic", "generic")
        prod_max = float(production["max_abs"])
        generic_max = float(generic["max_abs"])
        lines.append(
            "\t".join(
                [
                    label,
                    production["max_abs"],
                    production["rel_l2"],
                    generic["max_abs"],
                ]
            )
        )
        if prod_max > args.tolerance or generic_max > args.tolerance:
            failures += 1

    lines.append("")
    if labels and failures == 0:
        lines.append("VERDICT=REMAINING_Q8K_MATVECS_MATCH")
        lines.append(
            "All recurrent/attention output and MLP projections match the "
            f"captured TurboQuant Q8_K contract within {args.tolerance:g}."
        )
    else:
        lines.append("VERDICT=REMAINING_Q8K_MATVECS_MISMATCH")
        lines.append(f"job_count={len(labels)}")
        lines.append(f"failure_count={failures}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    return 0 if labels and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
