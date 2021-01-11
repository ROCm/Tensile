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
Usage: ${0} WORKING_PATH LOG_PATH OUTPUT_SUFFIX.yaml LIBRARY [options]

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
  -i | --id ID                    ?? 
Options for config generation:
  -a | --tile-aware               ?? 
  -m | --mfma                     Use MFMA instruction in tuning
  -r | --rk                       ?? Something with replacement kernels
  -s | --disable-strides          ?? Disables something
       --initialization           ?? Data initialization when tuning
       --problem-definition \\
             {gemm|batch|both}    ?? Which problems?
       --client {new|old|both}    Which client to use
"

OPTS=`getopt -o h,n:,p:,f:,b:,c:,t:,i:,a,m,r,s \
--long help,network:,tensile-path:,tensile-fork:,branch:,commit:,tag:,id:,\
tile-aware,mfma,rk,disable-strides,initialization:,problem-definition:,client \
 -n "${0}" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options" >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )            HELP=true; shift ;;
    -n | --network )         NETWORK=${2}; shift 2;;
    -p | --tensile-path )    TENSILE_PATH=${2}; shift 2;;
    -f | --tensile-fork )    TENSILE_FORK=${2}; shift 2;;
    -b | --branch  )         TENSILE_BRANCH=${2}; shift 2;;
    -c | --commit )          COMMIT=${2}; shift 2;;
    -t | --tag )             TAG=${2}; shift 2;;
    -i | --id )              ID=${2}; shift 2;;
    -a | --tile-aware )      TILE_AWARE=true; shift;;
    -m | --mfma )            MFMA=true; shift;;
    -r | --rk )              RK=true; shift;;
    -s | --disable-strides ) DISABLE_STRIDES=true; shift;;
    --initialization )       INITIALIZATION=${2}; shift;;
    --problem-definition )   PROBLEM_DEFINITION=${2}; shift;;
    --client )               TENSILE_CLIENT=${2}; shift 2;;
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
OUTPUT_SUFFIX=${3} # TODO: this is not really used in the rest of the tuning pipeline
LIBRARY=${4}

# TODO: test options have valid values

if [[ ${TENSILE_CLIENT} != both && ${TENSILE_CLIENT} != old ]]; then
  TENSILE_CLIENT=new
fi

# determine full path of tools root
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

# create configs for sizes in log file/dir
echo "Generating tuning configurations"
if [ -z ${NETWORK+x} ]; then
  EXTRACT_EXE="python3 ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${LOG_PATH} ${WORKING_PATH} ${OUTPUT_SUFFIX} ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION} ${INITIALIZATION} ${TENSILE_CLIENT} ${DISABLE_HPA}"
else
  EXTRACT_EXE="python3 ${AUTOMATION_ROOT}/GenerateTuningConfigurations.py ${LOG_PATH} ${NETWORK} ${WORKING_PATH} ${OUTPUT_SUFFIX} ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION} ${INITIALIZATION} ${TENSILE_CLIENT} ${DISABLE_HPA}"
fi
${EXTRACT_EXE}

# make outputed scripts executable
pushd ${OUT_SCRIPT_ROOT} > /dev/null
chmod +x *
popd > /dev/null

pushd ${OUT_SCRIPT2_ROOT} > /dev/null
chmod +x *
popd > /dev/null

# provision tensile if path not provided
if [ -z ${TENSILE_PATH+x} ]; then
  echo "Tensile path not provided. Trying to provision copy"
  TENSILE_PATH=`( cd "${WORKING_PATH}" && pwd )`/tensile/Tensile
  if [ ! -d ${TENSILE_PATH} ]; then
    provision_tensile
  else
    echo "Path already exists. Assuming Tensile previously provisioned"
  fi
else
  echo "Using existing Tensile path"
  TENSILE_PATH=`( cd "${TENSILE_PATH}" && pwd )`
fi

echo "Preparing scripts to run tuning"
ls ${BUILD_ROOT}/*.yaml | xargs -n1 basename | xargs ${SCRIPT_ROOT}/stage_tuning.sh ${BUILD_ROOT} ${STAGE_ROOT} ${TENSILE_PATH}
