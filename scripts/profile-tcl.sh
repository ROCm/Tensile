#!/bin/bash

# Variables
build_id=""
branch="develop"
arch="gfx900"
compiler="amdclang++"

# Constants
base_image="compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang"

# declare -a jobs=("16" "32")
# declare -a os_tags=("-ubuntu-24.04-stg1" "-ubuntu-22.04-stg1" "-ubuntu-20.04-stg1" "-rhel-9.x-stg1" "-sles-stg1")
declare -a jobs=("16")
declare -a os_tags=("-ubuntu-24.04-stg1")


usage() {
    echo "Run grid-based profiling analysis for TensileCreateLibrary under variable inputs"
    echo ""
    echo "Usage: $0 --build-id=<build-id> [--branch=<branch>] [--arch=<arch>] [--compiler=<compiler>]"
    echo ""
    echo "Parameters:"
    echo "  --build-id: The target docker build ID"
    echo "  --branch: The target branch [default: develop]"
    echo "  --arch: Target Gfx architecture(s) [default: gfx900]"
    echo "  --compiler: HIP-enabled compiler (must be in PATH) [default: amdclang++]"
    echo ""
    echo "Example:"
    echo "  $0 --build-id=12345 --branch=develop --arch='gfx90a'"
    echo ""
    echo "Dependencies:"
    echo "  docker: Docker is implicitly called and may install images"
}

find_tensile() {
  local query="*/Tensile/setup.py"
  local os_tag=$1
  # If several Tensile projects are found, use the first one
  docker run \
    --rm \
    --volume="$HOME:/mnt/host" \
    "$base_image:$build_id$os_tag" bash -c "find /mnt/host -path $query -exec dirname {} \; 2>&1 | head -n1"
}

find_logic() {
  local query="*/Tensile/Logic/asm_full"
  local os_tag=$1
  docker run \
    --rm \
    --volume="$HOME:/mnt/host" \
    "$base_image:$build_id$os_tag" bash -c "find /mnt/host -path $query"
}

run_suite() {

    for tag in "${os_tags[@]}"; do
        echo "> In container: $build_id$tag..."
        local tensile_path=$(find_tensile $tag)
        local logic_path=$(find_logic $tag)
        echo "    using Tensile: $tensile_path"
        echo "    using logic files: $logic_path"
        for n in "${jobs[@]}"; do
            docker run --rm --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri \
              --group-add=video --volume="$HOME:/mnt/host" "$base_image:$build_id$tag" \
              /bin/bash -c "$tensile_path/scripts/run-tcl.sh \
                --tensile-path=$tensile_path --logic-path=$logic_path --jobs=$n --arch=$arch --compiler=$compiler"
        done 
    done 
}

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --build-id=*) build_id="${arg#*=}" ;;
        --branch=*) branch="${arg#*=}" ;;
        --arch=*) arch="${arg#*=}" ;;
        --compiler=*) compiler="${arg#*=}" ;;
        --help) usage; exit 0 ;;
        *) echo "Invalid option: $arg"; usage; exit 1 ;;
    esac
done

# Check if all parameters are provided
if [ -z "$build_id" ] || [ -z "$branch" ] || [ -z "$arch" ] || [ -z "$compiler" ]; then
    usage 
    exit 1
fi

echo "> Profiling..."
echo "    build number:    $build_id"
echo "    branch:          $branch"
echo "    architecture(s): $arch"
echo "    compiler:        $compiler"


run_suite

