#!/bin/bash


function provision_rocblas() {

  local ROCBLAS_ID=$1 

  local PROVISION_ROCBLAS="${SCRIPT_ROOT}/provision_repo.sh -r -w ${ROCBLAS_ROOT} -b ${ROCBLAS_BRANCH} -i ${ROCBLAS_ID} --rocblas-fork ${ROCBLAS_FORK}"
  
  if [ -n "${ROCBLAS_FORK}" ]; then
    PROVISION_ROCBLAS="${PROVISION_ROCBLAS} --rocblas-fork ${ROCBLAS_FORK}"
  fi
  
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
  
  ${PROVISION_ROCBLAS}
}


HELP_STR="usage: ./provision_verification.sh [-w|--working-path <path>] [-r <Tensile reference>] [-b|--branch <branch>] [-c | --commit <github commit id>] [-t|--tag <githup tag>] [-l|--library <gpu library>] [-n|--no-massage]  [--rocblas-fork <username>] [-h|--help]"

HELP=false
ROCBLAS_BRANCH='develop'
ROCBLAS_FORK='RocmSoftwarePlatform'
MASSAGE=true
MERGE=true

OPTS=`getopt -o ht:w:b:c:i:r:nl: --long help,working-path:,size-log,output:,tag:,branch:,commit:,no-merge,no-massage,library:,type:,rocblas-fork: -n 'parse-options' -- "$@"`

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
    -n | --no-merge )     MERGE=false; shift ;;
    --no-massage )        MASSAGE=false; shift ;;
    -l | --library )      LIBRARY="$2"; shift 2;;
    --rocblas-fork )      ROCBLAS_FORK="$2"; shift 2;;
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

if [ -z ${LIBRARY+foo} ]; then
   printf "GPU Library not specified, assuming Vega 20\n"
   LIBRARY=vega20
fi

if [[ ${MERGE} == false ]]; then
   MASSAGE=false
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
MASSAGE_PATH="${LIBRARY_ROOT}/massage"
TENSILE_LIBRARY_PATH="${LIBRARY_ROOT}/tensile_library"

mkdir -p ${ROCBLAS_ROOT}
mkdir -p ${EXACT_PATH}
mkdir -p ${ASM_PATH}
mkdir -p ${ARCHIVE_PATH}

provision_rocblas reference

LOGIC_FILE_PATHS=`find ${TENSILE_PATH} -name 3_LibraryLogic | xargs -I{} printf "%s\n" {}`

REFERENCE_NAME=${ROCBLAS_ROOT}/rocBLAS-reference
REFERENCE_LIBRARY_ASM=${REFERENCE_NAME}/library/src/blas3/Tensile/Logic/asm_full
REFERENCE_LIBRARY_ARCHIVE=${REFERENCE_NAME}/library/src/blas3/Tensile/Logic/archive

for PATH_NAME in $LOGIC_FILE_PATHS; do
    cp ${PATH_NAME}/* ${EXACT_PATH}
done

cp ${REFERENCE_LIBRARY_ASM}/* ${ASM_PATH}
cp ${REFERENCE_LIBRARY_ARCHIVE}/* ${ARCHIVE_PATH}
cp ${ARCHIVE_PATH}/*yaml ${ASM_PATH}

MERGE_SCRIPT=${TENSILE_PATH}/Tensile/Utilities/merge_rocblas_yaml_files.py
MASSAGE_SCRIPT=${REFERENCE_LIBRARY_ARCHIVE}/massage.py

if [[ ${MERGE} == true ]]; then
  mkdir -p ${MERGE_PATH}
  mkdir -p ${MASSAGE_PATH}
  EXE_MERGE="python ${MERGE_SCRIPT} ${ASM_PATH} ${EXACT_PATH} ${MERGE_PATH}"
  ${EXE_MERGE}
else
  MERGE_PATH=${EXACT_PATH} 
fi

if [[ ${MASSAGE} == true ]]; then
  python ${MASSAGE_SCRIPT} ${MASSAGE_PATH} ${REFERENCE_LIBRARY_ASM}
fi

BUILD_ROCBLAS="./install.sh -c"

pushd ${REFERENCE_NAME} > /dev/null
${BUILD_ROCBLAS} > build-reference.out 2>&1
popd > /dev/null

TENSILE_CREATE_LIBRARY="${TENSILE_PATH}/Tensile/bin/TensileCreateLibrary --no-merge-files --no-legacy-components --no-short-file-names --no-library-print-debug --code-object-version=V2 --cxx-compiler=hcc ${MERGE_PATH} ${TENSILE_LIBRARY_PATH} HIP"
${TENSILE_CREATE_LIBRARY}
