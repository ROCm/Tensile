#!/bin/bash

# #################################################
# Parameter parsing
# #################################################

WORKING_PATH='release'
HELP=false
ROCBLAS=false
PROVISION_BRANCH='develop'
TENSILE_FORK='RocmSoftwarePlatform'
ROCBLAS_FORK='RocmSoftwarePlatform'

GIT_HOST="${TENSILE_HOST}"
PROVISION_PATH='Tensile'

HELP_STR="usage: $0 [-b|--branch <branch>] [-f|--tensile-fork <username>] [-w|--working-path <path>] [-i <identifier>] [-t|--tag <githup tag>] [-h|--help]"

OPTS=`getopt -o hf:rw:t:b:c:i: --long help,tensile-fork:,rocblas-fork:,working-path:,tag:,branch:,commit: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )         HELP=true; shift ;;
    -f | --tensile-fork ) TENSILE_FORK="$2"; shift 2;;
    --rocblas-fork )      ROCBLAS_FORK="$2"; shift 2;;
    -r )                  PROVISION_PATH='rocBLAS'; shift;;
    -w | --working-path ) WORKING_PATH="$2"; shift 2;;
    -t | --tag )          TAG="$2"; shift 3;;
    -b | --branch  )      PROVISION_BRANCH="$2"; shift 2;;
    -c | --commit )       COMMIT="$2"; shift 2;;
    -i )                  ID="$2"; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if $HELP; then
  echo "${HELP_STR}" >&2
  exit 2
fi

mkdir -p ${WORKING_PATH}

pushd ${WORKING_PATH} > /dev/null

TENSILE_HOST="https://github.com/${TENSILE_FORK}/Tensile.git"
ROCBLAS_HOST="https://github.com/${ROCBLAS_FORK}/rocBLAS-internal.git"

if [[ $PROVISION_PATH == "rocBLAS" ]]; then
  GIT_HOST="${ROCBLAS_HOST}"
else
  GIT_HOST="${TENSILE_HOST}"
fi

if [ -n "${ID+foo}" ]; then
  PROVISION_PATH="${PROVISION_PATH}-${ID}"
fi

if [ -n "${TAG}" ]; then
  PROVISION_PATH="${PROVISION_PATH}-${TAG}"
  cmd="git clone ${GIT_HOST} ${PROVISION_PATH}"
  ${cmd}
  pushd ${PROVISION_PATH} > /dev/null
  git checkout tags/${TAG}
  popd > /dev/null
else
  if [ -n "${COMMIT}" ]; then
    PROVISION_PATH="${PROVISION_PATH}-${COMMIT}"
  fi
  cmd="git clone -b ${PROVISION_BRANCH} ${GIT_HOST} ${PROVISION_PATH}"
  ${cmd}
  if [ -n "${COMMIT}" ]; then
    pushd ${PROVISION_PATH} > /dev/null
    git reset --hard ${COMMIT}
    popd > /dev/null
  fi
fi


popd > /dev/null


