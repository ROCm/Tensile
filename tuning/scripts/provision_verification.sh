#!/bin/bash


function provision_rocblas() {

  local ROCBLAS_ID=$1 

  local PROVISION_ROCBLAS="${SCRIPT_ROOT}/provision_repo.sh -r -w ${ROCBLAS_ROOT} -b ${ROCBLAS_BRANCH} -i ${ROCBLAS_ID}"

  if [ -n "${ID}" ]; then
    ROCBLAS_PATH="${ROCBLAS_PATH}-${ID}"
    PROVISION_ROCBLAS="${PROVISION_ROCBLAS} -i ${ID}"
  fi

  if [ -n "${TAG}" ]; then
    PROVISION_ROCBLAS="${PROVISION_ROCBLAS} -t ${TAG}"
  fi

  if [ -n "${COMMIT}" ]; then
    PROVISION_ROCBLAS="${PROVISION_ROCBLAS} -c ${COMMIT}"
  fi

  if [ -n "${ID}" ]; then
    PROVISION_ROCBLAS="${PROVISION_ROCBLAS} -i ${ID}"
  fi

  ${PROVISION_ROCBLAS}
}


HELP_STR="usage: ./provision_verification.sh [-w|--working-path <path>] [-r <Tensile reference>] [-b|--branch <branch>] [-c | --commit <github commit id>] [-t|--tag <githup tag>]  [-h|--help]"

HELP=false
ROCBLAS_BRANCH='develop'



OPTS=`getopt -o ht:w:b:c:i:r: --long help,working-path:,size-log,output:,tag:,branch:,commit:,library:,type: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )         HELP=true; shift ;;
    -w | --working-path ) WORKING_PATH="$2"; shift 2;;
    -t | --tag )          TAG="$2"; shift 3;;
    -b | --branch  )      ROCBLAS_BRANCH="$2"; shift 2;;
    -c | --commit )       COMMIT="$2"; shift 2;;
    -i )                  ID="$2"; shift 2;;
    -r )                  TENSILE_PATH="$2"; shift 2;;
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

if [ -z ${TENSILE_PATH+foo} ]; then
   printf "The tensile path is required\n"
   exit 2
fi


#determing full path of tools root
TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`

ROCBLAS_ROOT="${WORKING_PATH}/rocblas"
SCRIPT_ROOT="${TOOLS_ROOT}/scripts"
LIBRARY_ROOT="${WORKING_PATH}/library"
EXACT_PATH="${LIBRARY_ROOT}/exact"
MERGE_PATH="${LIBRARY_ROOT}/merge"
ASM_PATH="${LIBRARY_ROOT}/asm_full"
ARCHIVE_PATH="${LIBRARY_ROOT}/archive"


mkdir -p ${ROCBLAS_ROOT}
mkdir -p ${EXACT_PATH}
mkdir -p ${ASM_PATH}
mkdir -p ${ARCHIVE_PATH}
mkdir -p ${MERGE_PATH}

provision_rocblas reference
provision_rocblas verify

LOGIC_FILE_PATHS=`find ${TENSILE_PATH} -name 3_LibraryLogic | xargs -I{} printf "%s\n" {}`

REFERENCE_NAME=${ROCBLAS_ROOT}/rocBLAS-reference
VERIFY_NAME=${ROCBLAS_ROOT}/rocBLAS-verify
VERIFY_LIBRARY_ASM=${VERIFY_NAME}/library/src/blas3/Tensile/Logic/asm_full
VERIFY_LIBRARY_ARCHIVE=${VERIFY_NAME}/library/src/blas3/Tensile/Logic/archive

for PATH_NAME in $LOGIC_FILE_PATHS; do
    cp ${PATH_NAME}/* ${EXACT_PATH}
done

cp ${VERIFY_LIBRARY_ASM}/* ${ASM_PATH}
cp ${VERIFY_LIBRARY_ARCHIVE}/* ${ARCHIVE_PATH}
cp ${ARCHIVE_PATH}/*yaml ${ASM_PATH}

MERGE_SCRIPT=${TENSILE_PATH}/Tensile/Utilities/merge_rocblas_yaml_files.py
MESSAGE_SCRIPT=../archive/massage.py

EXE_MERGE="python ${MERGE_SCRIPT} ${ASM_PATH} ${EXACT_PATH} ${MERGE_PATH}"
${EXE_MERGE}

cp ${MERGE_PATH}/* ${VERIFY_LIBRARY_ASM}
cp ${MERGE_PATH}/vega20*{SB,DB}* ${ARCHIVE_PATH}
cp ${ARCHIVE_PATH}/*yaml ${VERIFY_LIBRARY_ASM}
cp ${ARCHIVE_PATH}/*yaml ${VERIFY_LIBRARY_ARCHIVE}

pushd ${VERIFY_LIBRARY_ASM} > /dev/null
python ${MESSAGE_SCRIPT}
popd > /dev/null


BUILD_ROCBLAS="./install.sh -c"

pushd ${REFERENCE_NAME} > /dev/null
${BUILD_ROCBLAS} > build-reference.out 2>&1
popd > /dev/null

pushd ${VERIFY_NAME} > /dev/null
${BUILD_ROCBLAS} > build-verify.out 2>&1
popd > /dev/null


