#!/bin/bash

function provision_rocblas() {

  local PROVISION_ROCBLAS="${SCRIPT_ROOT}/provision_repo.sh -r -w ${ROCBLAS_ROOT} -b ${ROCBLAS_BRANCH} -f ${ROCBLAS_FORK}"

  if [ -n "${ROCBLAS_FORK}" ]; then
    PROVISION_ROCBLAS="${PROVISION_ROCBLAS} -f ${ROCBLAS_FORK}"
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

HELP=false
ROCBLAS_BRANCH='develop'
ROCBLAS_FORK='RocmSoftwarePlatform'
MASSAGE=true
MERGE=true
REDO=false

HELP_STR="
Usage: ${0} WORKING_PATH TENSILE_PATH LIBRARY [options]

  where LIBRARY is {vega10|vega20|...}

Options:
  -h | --help                   Display this help message
  -p | --rocblas-path PATH      Path to existing rocBLAS
  -f | --rocblas-fork USERNAME  rocBLAS fork to use
  -b | --branch BRANCH          rocBLAS branch to use
  -c | --commit COMMIT_ID       rocBLAS commit to use
  -t | --tag GITHUP_TAG         rocBLAS tag to use
  -i | --id ID                  ??
  -s | --sclk                   Freq
  -n | --no-merge               Skip merge step
       --no-massage             Skip massage step
       --log-dir                Dir for logs
       --redo                   Force logic preparation, merge, massage, and library build steps to be redone
"

OPTS=`getopt -o h,p:,f:,b:,c:,t:,i:,s:,n \
--long help,rocblas-path:,rocblas-fork:,branch:,commit:,tag:,id:,sclk:,no-merge,no-massage,redo \
-n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options" >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )            HELP=true; shift ;;
    -p | --rocblas-path )    ROCBLAS_PATH=${2}; shift 2;;
    -f | --rocblas-fork )    ROCBLAS_FORK=${2}; shift 2;;
    -b | --branch  )         ROCBLAS_BRANCH=${2}; shift 2;;
    -c | --commit )          COMMIT=${2}; shift 2;;
    -t | --tag )             TAG=${2}; shift 2;;
    -i | --id)               ID=${2}; shift 2;;
    -s | --sclk )            SCLK=${2}; shift 2;;
    -n | --no-merge )        MERGE=false; shift ;;
    --no-massage )           MASSAGE=false; shift ;;
    --log-dir )              LOGS=${2}; shift 2;;
    --redo )                 REDO=true; shift 1;;
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
OUTPUT_SUFFIX=${2}
TENSILE_PATH=${3}
LIBRARY=${4}

# TODO: test options have valid values

if [ -z ${LOGS+x} ]; then
  LOGS=${WORKING_PATH}/logs
fi

if [[ ${MERGE} == false ]]; then
   MASSAGE=false
fi

# determine full path of tools root
TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`

ROCBLAS_ROOT=${WORKING_PATH}/rocblas
DEVTOOLS_ROOT=${WORKING_PATH}/devtools
SCRIPT_ROOT=${TOOLS_ROOT}/scripts
LIBRARY_ROOT=${WORKING_PATH}/library
EXACT_PATH=${LIBRARY_ROOT}/exact
MERGE_PATH=${LIBRARY_ROOT}/merge
ASM_PATH=${LIBRARY_ROOT}/asm_full
ARCHIVE_PATH=${LIBRARY_ROOT}/archive
MASSAGE_PATH=${LIBRARY_ROOT}/massage

TENSILE_LIBRARY_PATH=${LIBRARY_ROOT}/tensile_library
mkdir -p ${TENSILE_LIBRARY_PATH}
TENSILE_LIBRARY_PATH=$(realpath ${TENSILE_LIBRARY_PATH})

mkdir -p ${ROCBLAS_ROOT}
mkdir -p ${EXACT_PATH}
mkdir -p ${ASM_PATH}
mkdir -p ${ARCHIVE_PATH}
mkdir -p ${LOGS}

# provision rocblas if path not provided
if [ -z ${ROCBLAS_PATH+x} ]; then
  echo "rocBLAS path not provided. Trying to provision copy"
  ROCBLAS_PATH=`( cd "${WORKING_PATH}" && pwd )`/rocblas/rocBLAS
  if [ ! -d ${ROCBLAS_PATH} ]; then
    provision_rocblas
  else
    echo "Path already exists. Assuming rocBLAS previously provisioned"
  fi
else
  echo "Using existing rocBLAS path"
  ROCBLAS_PATH=`( cd "${ROCBLAS_PATH}" && pwd )`
fi
echo ${ROCBLAS_PATH}

# TODO: filter here to only get ones from "this" tuning exercise? Or should a dir only be for one exercise anyway?
LOGIC_FILE_PATHS=`find ${WORKING_PATH}/make -name 3_LibraryLogic | xargs -I{} printf "%s\n" {}`

REFERENCE_LIBRARY_ASM=${ROCBLAS_PATH}/library/src/blas3/Tensile/Logic/asm_full
REFERENCE_LIBRARY_ARCHIVE=${ROCBLAS_PATH}/library/src/blas3/Tensile/Logic/archive

# copy library logic files
if [[ $(ls -A ${EXACT_PATH} | wc -c) -eq 0 || ${REDO} == true ]]; then
  for PATH_NAME in $LOGIC_FILE_PATHS; do
    cp ${PATH_NAME}/* ${EXACT_PATH}
  done

  cp ${REFERENCE_LIBRARY_ASM}/* ${ASM_PATH}
  cp ${REFERENCE_LIBRARY_ARCHIVE}/* ${ARCHIVE_PATH}
  cp ${ARCHIVE_PATH}/*yaml ${ASM_PATH}
else
  echo "Logic directory non-empty. Assuming file copy done previously"
  echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
fi

# convert to efficiency
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

MERGE_SCRIPT=${TENSILE_PATH}/Tensile/Utilities/merge.py
MASSAGE_SCRIPT=${REFERENCE_LIBRARY_ARCHIVE}/massage.py

# perform merge step
if [[ ${MERGE} == true ]]; then
  mkdir -p ${MERGE_PATH}
  mkdir -p ${MASSAGE_PATH}

  if [[ ${LIBRARY} != arcturus && ${MASSAGE} == true ]]; then
    ASM_PATH=${ARCHIVE_PATH}
  fi

  echo "Merging tuned logic with existing logic"
  if [[ $(ls -A ${MERGE_PATH} | wc -c) -eq 0 || ${REDO} == true ]]; then
    EXE_MERGE="python3 ${MERGE_SCRIPT} ${ASM_PATH} ${EXACT_PATH} ${MERGE_PATH}"
    ${EXE_MERGE} 2>&1 | tee ${LOGS}/log-merge-script
  else
    echo "Merge directory non-empty. Assuming merge done previously"
    echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
  fi
else
  echo "Skipping merge step"
  MERGE_PATH=${EXACT_PATH}
fi

# perform massage step
if [[ ${MASSAGE} == true ]]; then
  echo "Massaging logic"
  if [[ $(ls -A ${MASSAGE_PATH} | wc -c) -eq 0 || ${REDO} == true ]]; then
     python3 ${MASSAGE_SCRIPT} ${MERGE_PATH} ${MASSAGE_PATH} 2>&1 | tee ${LOGS}/log-massage-script
  else
    echo "Massage directory non-empty. Assuming massage done previously"
    echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
  fi
else
  echo "Skipping massage step"
fi

# build rocBLAS
echo "Building rocBLAS"
ROCBLAS_BENCH_PATH=${ROCBLAS_PATH}/build/release/clients/staging/rocblas-bench
if [ ! -f ${ROCBLAS_BENCH_PATH} ]; then
  rm -r -f ${ROCBLAS_PATH}/build
  BUILD_ROCBLAS="./install.sh -d -c"
  pushd ${ROCBLAS_PATH} > /dev/null
  ${BUILD_ROCBLAS} > build-reference.out 2>&1
  popd > /dev/null
else
  echo "rocBLAS already built: skipping"
fi

# TODO: way to set which Tensile to use for create library?
# TODO: get correct Python version
CREATE_LIBRARY_EXE=${ROCBLAS_PATH}/build/release/virtualenv/lib/python3.6/site-packages/Tensile/bin/TensileCreateLibrary
TENSILE_CREATE_LIBRARY="${CREATE_LIBRARY_EXE} --merge-files --no-legacy-components --no-short-file-names --no-library-print-debug --code-object-version=V3 --cxx-compiler=hipcc --library-format=msgpack ${MERGE_PATH} ${TENSILE_LIBRARY_PATH} HIP"

# create new (tuned) Tensile library
echo "Creating new Tensile library"
if [ -f ${CREATE_LIBRARY_EXE} ]; then
    if [[ ! -d ${TENSILE_LIBRARY_PATH}/library || ${REDO} == true ]]; then
        ${TENSILE_CREATE_LIBRARY}
    else
      echo "Path already exists. Assuming library already built"
      echo "Use --redo to force redoing previously done library prep/merge/massage/build steps"
    fi
else
  echo "Error: could not find TensileCreateLibrary script"
fi
