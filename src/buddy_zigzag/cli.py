"""Command-line entry points for preparation, search, and compilation."""

import argparse
import sys
from pathlib import Path

import yaml

from .codegen import ScheduleCompileRequest
from .hardware import GemminiHardwareExporter
from .pipeline import DsePipeline, PrepareRequest
from .search import SearchRequest
from .settings import Settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bridge Buddy workloads, Gemmini hardware, and ZigZag schedules")
    parser.add_argument("--env-file", type=Path, help="Settings file; defaults to .env in the current directory")
    commands = parser.add_subparsers(dest="command", required=True)

    workload = commands.add_parser("prepare-workload", help="Build a workload and prepare mapping constraints")
    source = workload.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", help="Named Torch workload in the cases YAML")
    source.add_argument("--mlir", type=Path, help="Saved, single-function Linalg tensor module")
    workload.add_argument("--cases", type=Path, help="Case definitions; defaults to packaged examples")
    workload.add_argument("--mapping", type=Path, help="Manual mapping override; otherwise generate with --snapshot")
    workload.add_argument("--snapshot", type=Path, help="Gemmini hardware JSON for automatic WS matmul constraints")
    workload.add_argument("--out", type=Path, help="Default: ARTIFACTS_DIR/<case or MLIR stem>")
    workload.add_argument("--skip-zigzag", action="store_true", help="Skip ZigZag parser compatibility checking")

    hardware = commands.add_parser("prepare-hardware", help="Convert resolved Gemmini parameters to ZigZag hardware")
    hardware.add_argument("--snapshot", type=Path, required=True, help="JSON produced by the Gemmini Scala exporter")
    hardware.add_argument(
        "--template", type=Path, help="Model/cost assumptions; defaults to the packaged FP32 WS model"
    )
    hardware.add_argument(
        "--out", type=Path, help="Output YAML; default: hw_config/hardware.yaml; also writes a manifest"
    )

    search = commands.add_parser("search", help="Run ZigZag mapping search and cost evaluation")
    search.add_argument(
        "--workload-dir", type=Path, required=True, help="Prepared workload with workload.yaml/mapping.yaml"
    )
    search.add_argument("--hardware", type=Path, required=True, help="ZigZag hardware YAML")
    search.add_argument("--mapping", type=Path, help="Override <workload-dir>/mapping.yaml")
    search.add_argument("--objective", choices=["energy", "latency", "EDP"], default="latency")
    search.add_argument("--spatial-candidates", type=int, default=1)
    search.add_argument("--lpf-limit", type=int, default=4)
    search.add_argument("--show-progress", action="store_true")
    search.add_argument("--out", type=Path, help="Default: <workload-dir>/zigzag")

    compile_command = commands.add_parser("compile", help="Apply a selected mapping and generate CPU or Gemmini code")
    compile_command.add_argument("--target", choices=["cpu", "gemmini"], default="cpu")
    compile_command.add_argument("--snapshot", type=Path, help="Resolved hardware JSON; required for --target gemmini")
    compile_command.add_argument(
        "--gemmini-tiles", type=int, nargs=3, metavar=("M", "N", "K"), help="Explicit array-aligned tile override"
    )
    compile_command.add_argument(
        "--workload-dir", type=Path, required=True, help="Prepared workload with 01_imported.mlir and manifest.json"
    )
    compile_command.add_argument("--result", type=Path, required=True, help="Per-layer ZigZag *_complete.json result")
    compile_command.add_argument("--out", type=Path, help="Default: <workload-dir>/lowered")
    compile_command.add_argument(
        "--serialize-spatial", action="store_true", help="Explicitly serialize spatial lanes for CPU verification"
    )
    compile_command.add_argument("--verify", action="store_true", help="Execute scheduled Linalg on CPU against Torch")
    compile_command.add_argument("--trials", type=int, default=3, help="Random native verification trials, default 3")
    args = parser.parse_args(argv)
    try:
        settings = Settings.load(args.env_file)
        if args.command == "prepare-hardware":
            output = args.out if args.out is not None else settings.project_dir / "hw_config/hardware.yaml"
            result = GemminiHardwareExporter().export(args.snapshot, output, template_path=args.template)
            print(f"Validated ZigZag hardware: {result.hardware_path}")
            print(f"Resolved parameters and modeling assumptions: {result.manifest_path}")
        else:
            pipeline = DsePipeline(settings)
            if args.command == "prepare-workload":
                result = pipeline.prepare(
                    PrepareRequest(
                        case=args.case,
                        mlir=args.mlir,
                        cases=args.cases,
                        mapping=args.mapping,
                        snapshot=args.snapshot,
                        output_dir=args.out,
                        skip_zigzag=args.skip_zigzag,
                    )
                )
                print(f"Prepared and validated: {result.output_dir}")
                print(f"ZigZag workload: {result.output_dir / 'workload.yaml'}")
            elif args.command == "search":
                result = pipeline.search(
                    SearchRequest(
                        workload=args.workload_dir / "workload.yaml",
                        hardware=args.hardware,
                        mapping=args.mapping if args.mapping is not None else args.workload_dir / "mapping.yaml",
                        output_dir=args.out,
                        objective=args.objective,
                        spatial_candidates=args.spatial_candidates,
                        lpf_limit=args.lpf_limit,
                        show_progress=args.show_progress,
                    )
                )
                print(f"Search results: {result.output_dir}")
                print(f"Estimated energy: {result.energy}; latency: {result.latency}")
                for mapping in result.mappings:
                    print(f"Selected layer mapping: {mapping}")
            else:
                result = pipeline.compile(
                    ScheduleCompileRequest(
                        workload_dir=args.workload_dir,
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
                print(f"Compiled {args.target} schedule: {result.output_dir}")
                print(f"Lowered code: {result.output_dir / '06_lowered.ll'}")
                print(f"Native verification: {result.verification['status']}")
    except (ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError, TypeError, KeyError) as exc:
        print(f"{args.command.capitalize()} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
