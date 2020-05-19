#!/bin/bash



function extract_sizes() {

  pushd "${WORKING_PATH}" > /dev/null
  local EXTRACT_SIZE_PATH=`pwd`
  popd > /dev/null

  EXTRACT_EXE="python ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${SIZE_LOG} ${EXTRACT_SIZE_PATH} ${OUTPUT_FILE} ${LIBRARY} ${TILE_AWARE}"

  ${EXTRACT_EXE}

  pushd ${PEFORMANCE_PATH} > /dev/null
  chmod +x *
  popd > /dev/null
}

function extract_network_sizes() {

  pushd "${WORKING_PATH}" > /dev/null
  local EXTRACT_SIZE_PATH=`pwd`
  popd > /dev/null

  EXTRACT_EXE="python ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${SIZE_DIR} ${NETWORK} ${EXTRACT_SIZE_PATH} ${OUTPUT_FILE} ${LIBRARY} ${TILE_AWARE}"

  ${EXTRACT_EXE}

  pushd ${PEFORMANCE_PATH} > /dev/null
  chmod +x *
  popd > /dev/null
}

function build_configs() {
  # build the tuning configurations

  local HEADER_FILE="${CONFIGURATION_ROOT}/boiler/header.yml"
  local GENERATE_CONFIGURATION="${AUTOMATION_ROOT}/GenerateTuningConfigurations.py"

  BUILD_CONFIGS="python ${GENERATE_CONFIGURATION} -w ${WORKING_PATH} -d ${HEADER_FILE} -c ${CONFIGURATION_ROOT} -p ${PROBLEM_SPEC} -o ${OUTPUT_FILE}"
  ${BUILD_CONFIGS}

}

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

HELP_STR="usage: $0 [-w|--working-path <path>] [-z | --size-log <logfile path>] [-f|--tensile-fork <username>] [-b|--branch <branch>] [-c <github commit id>] [-t|--tag <github tag>] [--rocblas-fork <username>] [-o|--output <configuration filename>] [-y | --type <data type>] [-l | --library <library/schedule>] [-n] [[-h|--help]"
HELP=false
SUPPRESS_TENSILE=false
TENSILE_FORK='ROCmSoftwarePlatform'
ROCBLAS_FORK='ROCmSoftwarePlatform'
TENSILE_BRANCH='develop'
TENSILE_HOST="https://github.com/${TENSILE_FORK}/Tensile.git"
TILE_AWARE=false

OPTS=`getopt -o hw:z:d:n:t:f:b:c:o:y:l:ai: --long help,working-path:,size-log:,log-dir:,tag:,tensile-fork:,rocblas-fork:,branch:,commit:,output:,type:,library:,tile-aware,no-tensile -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )         HELP=true; shift ;;
    -w | --working-path ) WORKING_PATH="$2"; shift 2;;
    -z | --size-log )     SIZE_LOG="$2"; shift 2;;
    -d | --log-dir )      SIZE_DIR="$2"; shift 2;;
    -n | --network )      NETWORK="$2"; shift 2;;
    -t | --tag )          TAG="$2"; shift 3;;
    -f | --tensile-fork)  TENSILE_FORK="$2"; shift 2;;
    --rocblas-fork)       ROCBLAS_FORK="$2"; shift 2;;
    -b | --branch  )      TENSILE_BRANCH="$2"; shift 2;;
    -c | --commit )       COMMIT="$2"; shift 2;;
    -o | --output )       OUTPUT_FILE="$2"; shift 2;;
    -y | --type )         CONFIGURATION_TYPE="$2"; shift 2;;
    -l | --library )      LIBRARY="$2"; shift 2;;
    -a | --tile-aware )   TILE_AWARE=true; shift;;
    --no-tensile )        SUPPRESS_TENSILE=true; shift;;
    -i )                  ID="$2"; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if $HELP; then
  echo "${HELP_STR}" >&2
  exit 2
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

if [ -z ${CONFIGURATION_TYPE+foo} ]; then
   printf "Need specify a configuration type\n"
   exit 2
fi

#determining the full path of tools root
TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`

TENSILE_ROOT="${WORKING_PATH}/reops"
BUILD_ROOT="${WORKING_PATH}/configs"
STAGE_ROOT="${WORKING_PATH}/make"
TENSILE_ROOT="${WORKING_PATH}/tensile"
CONFIGURATION_ROOT="${TOOLS_ROOT}/configuration"
AUTOMATION_ROOT="${TOOLS_ROOT}/automation"
SCRIPT_ROOT="${TOOLS_ROOT}/scripts"
SIZE_PATH="${WORKING_PATH}/sizes"
PROBLEM_SPEC="${WORKING_PATH}/problem_spec.csv"
PEFORMANCE_PATH="${WORKING_PATH}/scripts"

mkdir -p ${BUILD_ROOT}
mkdir -p ${STAGE_ROOT}
mkdir -p ${TENSILE_ROOT}
mkdir -p ${SIZE_PATH}
mkdir -p ${PEFORMANCE_PATH}


# extracts the sizes from the logs and generates the tuning configurations
if [ -z ${NETWORK+foo} ]; then
  extract_sizes
else
  extract_network_sizes
fi

ls ${BUILD_ROOT}/*.yaml | xargs -n1 basename | xargs ${SCRIPT_ROOT}/stage_tuning.sh ${BUILD_ROOT} ${STAGE_ROOT}

# if enabled, this will provision tensile and set it up for tuning
if ${SUPPRESS_TENSILE} ; then
  echo "Suppressing Tensile provisioning"
else
  provision_tensile
fi


