#!/bin/bash

HELP=false
WORKING_PATH='release'
PROVISION_BRANCH='develop'
FORK='RocmSoftwarePlatform'

GIT_HOST="${TENSILE_HOST}"
PROVISION_PATH='Tensile'

HELP_STR="
Usage: ${0} [options]

Options:
-h | --help                    Display this help message
-r                             Provision rocBLAS instead of Tensile
-w | --working-path PATH       Working path for tuning
-f | --tensile-fork USERNAME   Fork to use
-b | --branch BRANCH           Branch to use
-c | --commit COMMIT_ID        Commit to use
-t | --tag GITHUB_TAG          Tag to use
-i | --id ID                   ID to append to directory name
"

if ! OPTS=$(getopt -o h,r,w:,f:,b:,c:,t:,i: \
--long help,working-path:,fork:,branch:,commit:,tag:,id: -n 'parse-options' -- "$@")
then
  echo "Failed parsing options"
  exit 1
fi

eval set -- "${OPTS}"

while true; do
  case ${1} in
    -h | --help )         HELP=true; shift ;;
    -r )                  PROVISION_PATH='rocBLAS'; shift;;
    -w | --working-path ) WORKING_PATH=${2}; shift 2;;
    -f | --fork )         FORK=${2}; shift 2;;
    -b | --branch  )      PROVISION_BRANCH=${2}; shift 2;;
    -c | --commit )       COMMIT=${2}; shift 2;;
    -t | --tag )          TAG=${2}; shift 2;;
    -i | --id )           ID=${2}; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ${HELP}; then
  echo "${HELP_STR}"
  exit 0
fi

mkdir -p "${WORKING_PATH}"

pushd "${WORKING_PATH}" > /dev/null || exit

TENSILE_HOST="https://github.com/${FORK}/Tensile.git"
ROCBLAS_HOST="https://github.com/${FORK}/rocBLAS-internal.git"

if [ $PROVISION_PATH == "rocBLAS" ]; then
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
  pushd "${PROVISION_PATH}" > /dev/null || exit
  git checkout "tags/${TAG}"
  popd > /dev/null || exit
else
  if [ -n "${COMMIT}" ]; then
    PROVISION_PATH="${PROVISION_PATH}-${COMMIT}"
  fi
  cmd="git clone -b ${PROVISION_BRANCH} ${GIT_HOST} ${PROVISION_PATH}"
  ${cmd}
  if [ -n "${COMMIT}" ]; then
    pushd "${PROVISION_PATH}" > /dev/null || exit
    git reset --hard "${COMMIT}"
    popd > /dev/null || exit
  fi
fi

popd > /dev/null || exit
