"""Thin command-line adapter around ZigzagExporter."""

import argparse
import sys
from pathlib import Path

import yaml

from .exporter import ExportRequest, ZigzagExporter
from .hardware import GemminiHardwareExporter
from .importer import ImportRequest, ZigzagImporter
from .settings import Settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bridge Buddy workloads, Gemmini hardware, and ZigZag schedules")
    parser.add_argument("--env-file", type=Path, help="Settings file; defaults to .env in the current directory")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("export")
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", help="Named Torch workload in the cases YAML")
    source.add_argument("--mlir", type=Path, help="Export a saved, single-function Linalg tensor module")
    command.add_argument("--cases", type=Path, help="Case definitions; defaults to packaged examples")
    command.add_argument("--mapping", type=Path, help="Optional ZigZag mapping; defaults to baseline operand links")
    command.add_argument("--out", type=Path, help="Default: ARTIFACTS_DIR/<case or MLIR stem>")
    command.add_argument(
        "--skip-zigzag", action="store_true", help="Export without ZigZag parser compatibility checking"
    )
    reverse = commands.add_parser("import", help="Apply one ZigZag matmul schedule for CPU or Gemmini lowering")
    reverse.add_argument("--target", choices=["cpu", "gemmini"], default="cpu")
    reverse.add_argument("--snapshot", type=Path, help="Resolved Gemmini hardware JSON; required for --target gemmini")
    reverse.add_argument(
        "--gemmini-tiles", type=int, nargs=3, metavar=("M", "N", "K"), help="Explicit array-aligned tile override"
    )
    reverse.add_argument(
        "--export-dir", type=Path, required=True, help="Original export with imported.mlir and manifest.json"
    )
    reverse.add_argument("--result", type=Path, required=True, help="Per-layer ZigZag *_complete.json result")
    reverse.add_argument("--out", type=Path, help="Default: <export-dir>/importer")
    reverse.add_argument(
        "--serialize-spatial", action="store_true", help="Explicitly serialize spatial lanes for CPU verification"
    )
    reverse.add_argument("--verify", action="store_true", help="Execute scheduled Linalg on CPU and compare with Torch")
    reverse.add_argument("--trials", type=int, default=3, help="Random native verification trials, default 3")
    hardware = commands.add_parser("hardware-export", help="Convert resolved Gemmini parameter JSON to ZigZag hardware")
    hardware.add_argument("--snapshot", type=Path, required=True, help="JSON produced by the Gemmini Scala exporter")
    hardware.add_argument(
        "--template", type=Path, help="Model/cost assumptions; defaults to the packaged FP32 WS model"
    )
    hardware.add_argument(
        "--out", type=Path, required=True, help="Output hardware .yaml; also writes a provenance manifest"
    )
    args = parser.parse_args(argv)
    try:
        settings = Settings.load(args.env_file)
        if args.command == "hardware-export":
            result = GemminiHardwareExporter().export(args.snapshot, args.out, template_path=args.template)
            print(f"Validated ZigZag hardware: {result.hardware_path}")
            print(f"Resolved parameters and modeling assumptions: {result.manifest_path}")
        elif args.command == "import":
            imported = ZigzagImporter(settings).run(
                ImportRequest(
                    export_dir=args.export_dir,
                    result=args.result,
                    output_dir=args.out,
                    serialize_spatial=args.serialize_spatial,
                    verify=args.verify,
                    trials=args.trials,
                    target=args.target,
                    snapshot=args.snapshot,
                    gemmini_tiles=tuple(args.gemmini_tiles) if args.gemmini_tiles else None,
                )
            )
            print(f"Imported {args.target} schedule: {imported.output_dir}")
            print(f"Lowered code: {imported.output_dir / 'lowered.ll'}")
            print(f"Native verification: {imported.verification['status']}")
        else:
            exporter = ZigzagExporter(settings)
            result = exporter.export(
                ExportRequest(
                    case=args.case,
                    mlir=args.mlir,
                    cases=args.cases,
                    mapping=args.mapping,
                    output_dir=args.out,
                    skip_zigzag=args.skip_zigzag,
                )
            )
            print(f"Exported and validated: {result.output_dir}")
            print(f"ZigZag workload: {result.output_dir / 'workload.yaml'}")
    except (ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError, TypeError, KeyError) as exc:
        print(f"{args.command.capitalize()} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
