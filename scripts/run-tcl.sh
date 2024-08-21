#!/bin/bash

usage() {
    echo "Run TensileCreateLibrary with timestamped log and build directory"
    echo ""
    echo "Usage: $0 --tensile-path=<tensile-path> --logic-path=<logic-path> --jobs=<jobs> [--archs=<archs>] [--compiler=<compiler>]"
    echo ""
    echo "Parameters:"
    echo "  --tensile-path: Path to root directory of Tensile"
    echo "  --logic-path: Path to directory containing logic files"
    echo "  --jobs: Number of concurrent processes to use"
    echo "  --archs: Target Gfx architecture(s) [default: all]"
    echo "  --compiler: HIP-enabled compiler (must be in PATH) [default: amdclang++]"
    echo ""
    echo "Example:"
    echo "  $0 --tensile-path=/mnt/host/Tensile --logic-path=/mnt/host/Logic --jobs=16"
}

main() {
  export TENSILE_PROFILE=ON
  export PYTHONPATH="$tensile_path"
  /bin/bash -c "$tensile_path/Tensile/bin/TensileCreateLibrary $logic_path $build_dir $language \
     --merge-files \
     --separate-architecture \
     --lazy-library-loading \
     --no-short-file-names \
     --code-object-version=default \
     --cxx-compiler=$compiler \
     --jobs=$jobs \
     --library-format=msgpack \
     --architecture=$archs" > "$tensile_path/$log_file" 2>&1
}

# Variables
tensile_path=""
logic_path=""
jobs=""
archs="all"
compiler="amdclang++"

# Constants
language="HIP"
build_dir="build-tcl-$(date +'%Y-%m-%dT%H-%M-%S')"
log_file="tcl-profile-$(date +'%Y-%m-%dT%H-%M-%S').log"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --tensile-path=*) tensile_path="${arg#*=}" ;;
        --logic-path=*) logic_path="${arg#*=}" ;;
        --jobs=*) jobs="${arg#*=}" ;;
        --archs=*) archs="${arg#*=}" ;;
        --compiler=*) compiler="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

# Check if all parameters are provided
if [ -z "$tensile_path" ] || [ -z "$logic_path" ] || [ -x "$jobs" ] || [ -z "$archs" ] || [ -z "$compiler" ]; then
    usage
    exit 1
fi

main
