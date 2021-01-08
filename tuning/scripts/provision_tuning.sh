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
}

# main execution starts here

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

HELP_STR="
Usage: ${0} WORKING_PATH LOG_PATH OUTPUT_SUFFIX LIBRARY [options]

  where LIBRARY is {vega10|vega20|...}

Options:
  -h | --help                    Display this help message
  -n | --network NAME            Neural network name. If this is set, LOG_PATH should be a directory. Will only tune log files with this string in the file name
  -p | --tensile-path PATH       Path to existing Tensile (will not provision new copy)
Options for provisioning Tensile:
  -f | --tensile-fork USERNAME    Tensile fork to use
  -b | --branch BRANCH            Tensile branch to use
  -c | --commit COMMIT_ID         Tensile commit to use
  -t | --tag GITHUB_TAG           Tensile tag to use
       --id ID                    ?? 
Options for config generation:
  -a | --tile-aware               ?? 
  -m | --mfma                     Use MFMA instruction in tuning
  -r | --rk                       ?? Something with replacement kernels
  -s | --disable-strides          ?? Disables something
  -i | --initialization           ?? Data initialization when tuning
       --problem-definition \\
             {gemm|batch|both}    ?? Which problems?
       --client {new|old|both}    Which client to use
"

GET_OPT=`getopt -o h,n:,p:,f:,b:,c:,t:a,m,r,s,i: \
--long help,network:,tensile-path:,tensile-fork:,branch:,commit:,tag:,id:,\
tile-aware,mfma,rk,disable-strides,initialization:,problem-definition:,client \
 -n "${0}" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options" >&2 ; exit 1 ; fi

eval set -- "${GET_OPT}"

while true; do
  case "$1" in
    -h | --help )           HELP=true; shift ;;
    -n | --network )        NETWORK=${2}; shift 2;;
    -p | --tensile-path)    TENSILE_PATH=${2}; shift 2;;
    -f | --tensile-fork)    TENSILE_FORK=${2}; shift 2;;
    -b | --branch  )        TENSILE_BRANCH=${2}; shift 2;;
    -c | --commit )         COMMIT=${2}; shift 2;;
    -t | --tag )            TAG=${2}; shift 3;;
         --id )             ID=${2}; shift 2;;
    -a | --tile-aware )     TILE_AWARE=true; shift;;
    -m | --mfma )           MFMA=true; shift;;
    -r | --rk )             RK=true; shift;;
    -s | --disable-strides) DISABLE_STRIDES=true; shift;;
    -i | --initialization ) INITIALIZATION=${2}; shift;;
    --problem-definition )  PROBLEM_DEFINITION=${2}; shift;;
    --client )              TENSILE_CLIENT=${2}; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if $HELP; then
  echo "${HELP_STR}" >&2
  exit 0
fi

if [ $# != 4 ]; then
  echo "Exactly four positional args required"
  echo "See ${0} --help"
  exit 2
fi

WORKING_PATH=${1}
LOG_PATH=${2}
OUTPUT_SUFFIX=${3}
LIBRARY=${4}

# TODO: test library, init, client, and problem values are valid

if [[ ${TENSILE_CLIENT} != both && ${TENSILE_CLIENT} != old ]]; then
  echo "Setting Tensile Client to new"
  TENSILE_CLIENT=new
fi

# determining the full path of tools root
TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`

BUILD_ROOT=${WORKING_PATH}/configs
STAGE_ROOT=${WORKING_PATH}/make
OUT_SCRIPT_ROOT=${WORKING_PATH}/scripts
OUT_SCRIPT2_ROOT=${WORKING_PATH}/scripts2
TENSILE_ROOT=${WORKING_PATH}/tensile
AUTOMATION_ROOT=${TOOLS_ROOT}/automation
SCRIPT_ROOT=${TOOLS_ROOT}/scripts

mkdir -p ${STAGE_ROOT}

# extracts the sizes from the logs and generates the tuning configurations
if [ -z ${NETWORK+x} ]; then
  EXTRACT_EXE="python3 ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${LOG_PATH} ${WORKING_PATH} ${OUTPUT_SUFFIX} ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION}  ${INITIALIZATION} ${TENSILE_CLIENT} ${DISABLE_HPA}"
else
  EXTRACT_EXE="python3 ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${LOG_PATH} ${NETWORK} ${WORKING_PATH} ${OUTPUT_SUFFIX} ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION} ${INITIALIZATION} ${TENSILE_CLIENT} ${DISABLE_HPA}"
fi
${EXTRACT_EXE}

# make output scripts executable
pushd ${OUT_SCRIPT_ROOT} > /dev/null
chmod +x *
popd > /dev/null

pushd ${OUT_SCRIPT2_ROOT} > /dev/null
chmod +x *
popd > /dev/null

# provision tensile if path not provided
if [ -z ${TENSILE_PATH+x} ]; then
  provision_tensile
  TENSILE_PATH=`( cd "${WORKING_PATH}" && pwd )`/tensile/Tensile
else
  TENSILE_PATH=`( cd "${TENSILE_PATH}" && pwd )`
fi

# prepare scripts to run tuning
ls ${BUILD_ROOT}/*.yaml | xargs -n1 basename | xargs ${SCRIPT_ROOT}/stage_tuning.sh ${BUILD_ROOT} ${STAGE_ROOT} ${TENSILE_PATH}
