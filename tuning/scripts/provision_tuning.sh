#!/bin/bash

function provision_tensile() {

  local PROVISION_TENSILE="${SCRIPT_ROOT}/provision_repo.sh -w ${TENSILE_ROOT} -b ${TENSILE_BRANCH} -f ${TENSILE_FORK}"

  local TENSILE_PATH=Tensile

  if [ -n "${ID}" ]; then
    TENSILE_PATH="${TENSILE_PATH}-${ID}"
    PROVISION_TENSILE="${PROVISION_TENSILE} -i ${ID}"
  fi

  if [ -n "${TAG}" ]; then
    TENSILE_PATH="${TENSILE_PATH}-${TAG}"
  else
    if [ -n "${COMMIT}" ]; then
      TENSILE_PATH="${TENSILE_PATH}-${COMMIT}"
    fi
  fi

  if [ -n "${TAG}" ]; then
    PROVISION_TENSILE="${PROVISION_TENSILE} -t ${TAG}"
  fi

  if [ -n "${COMMIT}" ]; then
    PROVISION_TENSILE="${PROVISION_TENSILE} -c ${COMMIT}"
  fi

  ${PROVISION_TENSILE}

  cp -r ${STAGE_ROOT}/* ${TENSILE_ROOT}/${TENSILE_PATH}
}

HELP_STR="
Usage: ./provisions_tuning.sh -w WORKING_PATH {-z LOG_FILE | -d LOG_DIR -n NETWORK_NAME} -o OUTPUT_NAME -l LIBRARY [options]

Options:
  [-h|--help]                     Display this help message
  [-w|--working-path PATH]        Working path for tuning
  [-z|--size-log PATH]            Log file containing sizes to tune
  [-d|--log-dir PATH]             Directory containing log files
  [-n|--network NAME]             Neural network name. ?? Will only tune log files with this string in the file name
  [-o|--output NAME]              Output name to append to config files generated
  [-l|--library LIBRARY]          Library to tune on (e.g. vega20)
  [-p|--tensile-path PATH]        Path to existing Tensile (will not provision new copy)
Options for provisioning Tensile:
  [-f|--tensile-fork USERNAME]    Tensile fork to use
  [-b|--branch BRANCH]            Tensile branch to use
  [-c|--commit COMMIT_ID]         Tensile commit to use
  [-t|--tag GITHUB_TAG]           Tensile tag to use
  [--id ID]                       ?? 
  [--no-tensile]                  Skip provisioning Tensile
Options for config generation:
  [-a|--tile-aware]               ?? 
  [-m|--mfma]                     Use MFMA instruction in tuning
  [-r|--rk]                       ?? Something with replacement kernels
  [-s|--disable-strides]          ?? Disables something
  [-i|--initialization]           ?? Data initialization when tuning
  [--problem-definition \\
      {gemm|batch|both}]          ?? Which problems?
  [--client {new|old|both}]       Which client to use
"
HELP=false
TENSILE_CLIENT=new
SUPPRESS_TENSILE=false
TENSILE_FORK='ROCmSoftwarePlatform'
TENSILE_BRANCH='develop'
TENSILE_HOST="https://github.com/${TENSILE_FORK}/Tensile.git"
TILE_AWARE=false
MFMA=false
RK=false
DISABLE_STRIDES=false
PROBLEM_DEFINITION=both
INITIALIZATION=rand_int
DISABLE_HPA=false

OPTS=`getopt -o hw:z:d:n:t:f:p:b:c:o:l:amrsi: \
--long help,working-path:,size-log:,log-dir:,network:,tag:,tensile-fork:,\
rocblas-fork:,tensile-path:,branch:,commit:,output:,library:,client:,\
tile-aware,mfma,rk,problem-definition:,disable-strides,initialization:,\
no-tensile,id: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )           HELP=true; shift ;;
    -w | --working-path )   WORKING_PATH="$2"; shift 2;;
    -z | --size-log )       SIZE_LOG="$2"; shift 2;;
    -d | --log-dir )        SIZE_DIR="$2"; shift 2;;
    -n | --network )        NETWORK="$2"; shift 2;;
    --client )              TENSILE_CLIENT="$2"; shift 2;;
    -p | --tensile-path)    TENSILE_PATH="$2"; shift 2;;
    -f | --tensile-fork)    TENSILE_FORK="$2"; shift 2;;
    -b | --branch  )        TENSILE_BRANCH="$2"; shift 2;;
    -c | --commit )         COMMIT="$2"; shift 2;;
    -t | --tag )            TAG="$2"; shift 3;;
    -o | --output )         OUTPUT_FILE="$2"; shift 2;;
    -l | --library )        LIBRARY="$2"; shift 2;;
    -a | --tile-aware )     TILE_AWARE=true; shift;;
    -m | --mfma )           MFMA=true; shift;;
    -r | --rk )             RK=true; shift;;
    --problem-definition )  PROBLEM_DEFINITION="$2"; shift;;
    -s | --disable-strides) DISABLE_STRIDES=true; shift;;
    -i | --initialization ) INITIALIZATION="$2"; shift;;
    --no-tensile )          SUPPRESS_TENSILE=true; shift;;
    --id )                  ID="$2"; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if $HELP; then
  echo "${HELP_STR}" >&2
  exit 0
fi

if [ -z ${WORKING_PATH+foo} ]; then
  printf "A working path is required\n"
  exit 2
fi

if [ -z ${SIZE_LOG+foo} ] && [ -z ${SIZE_DIR+foo} ]; then
  printf "A problem specification file or directory is required\n"
  exit 2
fi

if [ -z ${OUTPUT_FILE+foo} ]; then
  printf "Need a configuration file name to generate\n"
  exit 2
fi
if [ -z ${LIBRARY+foo} ]; then
  printf "Need specify a target platform for tuning\n"
  exit 2
fi

if [[ "${TENSILE_CLIENT}" != both && "${TENSILE_CLIENT}" != old ]]; then
  printf "Setting Tensile Client to new\n"
  TENSILE_CLIENT=new
fi

# determining the full path of tools root
TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`

BUILD_ROOT="${WORKING_PATH}/configs"
STAGE_ROOT="${WORKING_PATH}/make"
OUT_SCRIPT_ROOT="${WORKING_PATH}/scripts"
OUT_SCRIPT2_ROOT="${WORKING_PATH}/scripts2"
TENSILE_ROOT="${WORKING_PATH}/tensile"
AUTOMATION_ROOT="${TOOLS_ROOT}/automation"
SCRIPT_ROOT="${TOOLS_ROOT}/scripts"

mkdir -p ${STAGE_ROOT}

# extracts the sizes from the logs and generates the tuning configurations
if [ -z ${NETWORK+foo} ]; then
  EXTRACT_EXE="python3 ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${SIZE_LOG} ${WORKING_PATH} ${OUTPUT_FILE} ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION}  ${INITIALIZATION} ${TENSILE_CLIENT} ${DISABLE_HPA}"
else
  EXTRACT_EXE="python3 ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${SIZE_DIR} ${NETWORK} ${WORKING_PATH} ${OUTPUT_FILE} ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION} ${INITIALIZATION} ${TENSILE_CLIENT} ${DISABLE_HPA}"
fi
${EXTRACT_EXE}

# make output scripts executable
pushd ${OUT_SCRIPT_ROOT} > /dev/null
chmod +x *
popd > /dev/null

pushd ${OUT_SCRIPT2_ROOT} > /dev/null
chmod +x *
popd > /dev/null

# prepare scripts to run tuning
ls ${BUILD_ROOT}/*.yaml | xargs -n1 basename | xargs ${SCRIPT_ROOT}/stage_tuning.sh ${BUILD_ROOT} ${STAGE_ROOT}

if [ -z ${TENSILE_PATH+foo} ]; then
  # tensile path not set: provision copy if not suppressed
  if ${SUPPRESS_TENSILE} ; then
    echo "Suppressing Tensile provisioning"
  else
    provision_tensile
  fi
  # use provided tensile path
else
  cp -r ${STAGE_ROOT}/* ${TENSILE_PATH}
fi
