#!/usr/bin/env bash
# Add our Scala exporter to this sbt session only; never edit Gemmini sources.
set -Eeuo pipefail

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ ${1:-} = --help || ${1:-} = -h ]]; then
  printf 'Usage: scripts/export_gemmini_config.sh [OUTPUT.json]\nExports FP32DefaultConfig without SoC elaboration.\n'
  exit 0
fi
(( $# <= 1 )) || die 'Please select just one output file. Try --help.'
output=${1:-hw_config/fp32_parameters.json}
[[ $output = /* ]] || output=$repo_root/$output
chipyard_dir=${CHIPYARD_DIR:-$repo_root/third_party/chipyard}
[[ $chipyard_dir = /* ]] || chipyard_dir=$repo_root/$chipyard_dir
[[ -f $chipyard_dir/env.sh ]] || die 'Please finish Chipyard setup first; env.sh is missing.'
if ! command -v conda >/dev/null; then
  [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]] || die 'Please activate your Chipyard environment first.'
  export PATH="$(dirname -- "$CONDA_EXE"):$PATH"
fi
trap 'die "Could not load Chipyard. Please check its env.sh."' ERR
set +u
# shellcheck source=/dev/null
source "$chipyard_dir/env.sh"
set -u
trap - ERR
command -v sbt >/dev/null || die 'sbt was not found. Please check your Chipyard environment.'
source_file=$repo_root/src/buddy_zigzag/hardware/scala/GemminiConfigExport.scala
# Escape paths for the Scala string and sbt command parser, not for the shell.
source_file=${source_file//\\/\\\\}
source_file=${source_file//\"/\\\"}
output=${output//\\/\\\\}
output=${output//\"/\\\"}
cd -- "$chipyard_dir"
sbt -batch \
  "set gemmini / Compile / unmanagedSources += file(\"$source_file\")" \
  "gemmini/runMain gemmini.GemminiConfigExportMain --output \"$output\"" \
  || die 'Hardware export failed. Please check the sbt output above.'
