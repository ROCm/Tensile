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


HELP_STR="usage: ./provision_verification.sh [-w|--working-path <path>] [-r <Tensile reference>] [-b|--branch <branch>] [-c | --commit <github commit id>] [-t|--tag <githup tag>] [-l|--library <gpu library>] [-n|--no-merge] [--no-massage]  [--rocblas-fork <username>] [-h|--help]"

HELP=false
ROCBLAS_BRANCH='develop'
ROCBLAS_FORK='RocmSoftwarePlatform'
MASSAGE=true
MERGE=true

OPTS=`getopt -o ht:w:b:c:i:r:nl:f: --long help,working-path:,size-log,output:,tag:,branch:,commit:,no-merge,no-massage,library:,--sclk:,type:,rocblas-fork: -n 'parse-options' -- "$@"`

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
    -f | --sclk )         SCLK="$2"; shift 2;;
    --rocblas-fork )      ROCBLAS_FORK="$2"; shift 2;;
    --log-dir )           LOGS="$2"; shift 2;;
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

if [ -z ${LOGS+foo} ]; then
  LOGS=${WORKING_PATH}/logs 
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
DEVTOOLS_ROOT="${WORKING_PATH}/devtools"
SCRIPT_ROOT="${TOOLS_ROOT}/scripts"
LIBRARY_ROOT="${WORKING_PATH}/library"
EXACT_PATH="${LIBRARY_ROOT}/exact"
MERGE_PATH="${LIBRARY_ROOT}/merge"
ASM_PATH="${LIBRARY_ROOT}/asm_full"
ARCHIVE_PATH="${LIBRARY_ROOT}/archive"
MASSAGE_PATH="${LIBRARY_ROOT}/massage"


TENSILE_LIBRARY_PATH="${LIBRARY_ROOT}/tensile_library"
mkdir -p ${TENSILE_LIBRARY_PATH}
TENSILE_LIBRARY_PATH=$(realpath ${TENSILE_LIBRARY_PATH})

mkdir -p ${ROCBLAS_ROOT}
mkdir -p ${EXACT_PATH}
mkdir -p ${ASM_PATH}
mkdir -p ${ARCHIVE_PATH}
mkdir -p ${LOGS}

ROCBLAS_REFERENCE_NAME="${ROCBLAS_ROOT}/rocBLAS-reference"

if [ ! -d ${ROCBLAS_REFERENCE_NAME} ]; then
    provision_rocblas reference
fi

LOGIC_FILE_PATHS=`find ${TENSILE_PATH} -name 3_LibraryLogic | xargs -I{} printf "%s\n" {}`

REFERENCE_NAME=${ROCBLAS_ROOT}/rocBLAS-reference
REFERENCE_LIBRARY_ASM=${REFERENCE_NAME}/library/src/blas3/Tensile/Logic/asm_full
REFERENCE_LIBRARY_ARCHIVE=${REFERENCE_NAME}/library/src/blas3/Tensile/Logic/archive

if [[ $(ls -A ${EXACT_PATH} | wc -c) -eq 0 ]]; then
  for PATH_NAME in $LOGIC_FILE_PATHS; do
    cp ${PATH_NAME}/* ${EXACT_PATH}
  done

  cp ${REFERENCE_LIBRARY_ASM}/* ${ASM_PATH}
  cp ${REFERENCE_LIBRARY_ARCHIVE}/* ${ARCHIVE_PATH}
  cp ${ARCHIVE_PATH}/*yaml ${ASM_PATH}
fi

MERGE_SCRIPT=${TENSILE_PATH}/Tensile/Utilities/merge.py
MASSAGE_SCRIPT=${REFERENCE_LIBRARY_ARCHIVE}/massage.py

if [ "${LIBRARY}" == arcturus ]; then
  if [ "${PUBLIC}" == false ]; then
    if [ ! -f ${LOGS}/log-efficiency ]; then
        pushd ${WORKING_PATH}  > /dev/null
        git clone https://github.com/RocmSoftwarePlatform/rocmdevtools.git -b efficiency
        python3 rocmdevtools/scripts/tuning/convertToEfficiency.py ${EXACT_PATH} ${LIBRARY} ${SCLK} 2>&1 | tee ${LOGS}/log-efficiency
        popd > /dev/null
    fi
  fi
fi

if [[ ${MERGE} == true ]]; then
  mkdir -p ${MERGE_PATH}
  mkdir -p ${MASSAGE_PATH}

  if [[ ${LIBRARY} != arcturus && ${MASSAGE} == true ]]; then
    ASM_PATH=${ARCHIVE_PATH}
  fi

  if [[ $(ls -A ${MERGE_PATH} | wc -c) -eq 0 ]]; then
    echo "merging exact logic"
    EXE_MERGE="python3 ${MERGE_SCRIPT} ${ASM_PATH} ${EXACT_PATH} ${MERGE_PATH}"
    ${EXE_MERGE} 2>&1 | tee ${LOGS}/log-merge-script
  fi
  
else
  MERGE_PATH=${EXACT_PATH}
fi

if [[ ${MASSAGE} == true ]]; then
  if [[ $(ls -A ${MASSAGE_PATH} | wc -c) -eq 0 ]]; then
     python3 ${MASSAGE_SCRIPT} ${MERGE_PATH} ${MASSAGE_PATH} 2>&1 | tee ${LOGS}/log-massage-script
  fi
fi

ROCBLAS_BENCH_PATH="${ROCBLAS_REFERENCE_NAME}/build/release/clients/staging/rocblas-bench"

if [ ! -f ${ROCBLAS_BENCH_PATH} ]; then
    rm -r -f ${ROCBLAS_REFERENCE_NAME}/build
    BUILD_ROCBLAS="./install.sh -c"
    pushd ${REFERENCE_NAME} > /dev/null
    ${BUILD_ROCBLAS} > build-reference.out 2>&1
    popd > /dev/null
fi

CREATE_LIBRARY_EXE=${REFERENCE_NAME}/build/release/virtualenv/lib/python3.6/site-packages/Tensile/bin/TensileCreateLibrary
TENSILE_CREATE_LIBRARY="${CREATE_LIBRARY_EXE} --merge-files --no-legacy-components --no-short-file-names --no-library-print-debug --code-object-version=V3 --cxx-compiler=hipcc --library-format=msgpack ${MERGE_PATH} ${TENSILE_LIBRARY_PATH} HIP"
if [ -f ${CREATE_LIBRARY_EXE} ]; then
    if [ ! -d ${TENSILE_LIBRARY_PATH}/library ]; then
        ${TENSILE_CREATE_LIBRARY}
    fi
fi
