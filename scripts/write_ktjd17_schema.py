#!/usr/bin/env python3
"""Build or validate the fail-closed KTJD-17 schema artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ktjd17.schema import (
    KTJD17_SOURCE_PLAN,
    KTJD17_SOURCE_PLAN_COMMIT,
    SchemaValidationError,
    build_schema,
    load_schema,
    write_schema,
)


def _object_json(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/schema.json"))
    parser.add_argument("--validate-only", type=Path)
    parser.add_argument("--expected-fps-target", type=float)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps-target", type=float, default=30.0,
                        help="Candidate until --frozen and train-only calibration evidence exist")
    parser.add_argument("--smoother-id")
    parser.add_argument("--smoother-params", type=_object_json, default=None)
    parser.add_argument("--short-clip-rule")
    parser.add_argument("--heading-eps-h", type=float)
    parser.add_argument("--contact-tau-h", type=float)
    parser.add_argument("--contact-tau-v", type=float)
    parser.add_argument("--normalization-gains", type=float, nargs=3)
    parser.add_argument("--j-max", type=int)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--source-plan", default=KTJD17_SOURCE_PLAN)
    parser.add_argument("--source-plan-commit", default=KTJD17_SOURCE_PLAN_COMMIT)
    parser.add_argument("--calibration-run-id", action="append", default=[])
    parser.add_argument("--train-split-protocol")
    parser.add_argument("--frozen-at-utc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only is not None:
        schema = load_schema(
            args.validate_only,
            expected_fps_target=args.expected_fps_target,
            require_frozen=args.require_frozen,
        )
        print(
            f"KTJD-17 schema valid: {args.validate_only.resolve()} "
            f"status={schema['calibration']['status']} fps_target={schema['fps_target']}"
        )
        return 0

    schema = build_schema(
        fps_target=args.fps_target,
        smoother_id=args.smoother_id,
        smoother_params=args.smoother_params,
        short_clip_rule=args.short_clip_rule,
        heading_eps_h=args.heading_eps_h,
        contact_tau_h=args.contact_tau_h,
        contact_tau_v=args.contact_tau_v,
        normalization_gains=args.normalization_gains,
        j_max=args.j_max,
        frozen=args.frozen,
        source_plan=args.source_plan,
        source_plan_commit=args.source_plan_commit,
        calibration_run_ids=args.calibration_run_id,
        train_split_protocol=args.train_split_protocol,
        frozen_at_utc=args.frozen_at_utc,
    )
    output = write_schema(schema, args.output, overwrite=args.overwrite)
    print(
        f"KTJD-17 schema written: {output.resolve()} "
        f"status={schema['calibration']['status']} fps_target={schema['fps_target']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SchemaValidationError, FileExistsError, FileNotFoundError) as exc:
        print(f"KTJD-17 schema error: {exc}", file=sys.stderr)
        raise SystemExit(2)
