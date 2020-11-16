#!/bin/bash


HELP_STR="usage: ./run_validation.sh [-w|--working-path <path>] [-s|--test-scripts <path to test yamls] [-i | id] [-h|--help]"                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                        
HELP=false                                                                                                                                                                                                                                                              
ROCBLAS_BRANCH='develop'                                                                                                                                                                                                                                                
ROCBLAS_FORK='RocmSoftwarePlatform'                                                                                                                                                                                                                                     
MASSAGE=true
MERGE=true

OPTS=`getopt -o ht:w:s:i: --long help,working-path:,test-scripts: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )         HELP=true; shift ;;
    -w | --working-path ) WORKING_PATH="$2"; shift 2;;
    -s | --test-scripts ) SCRIPTS_PATH="$2"; shift 2;;
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

if [ -z ${SCRIPTS_PATH+foo} ]; then
    printf "A script path is required\n"
    exit 2
fi

ROCBLAS_ROOT="${WORKING_PATH}/rocblas"
SCRIPT_ROOT="${SCRIPTS_PATH}"
LIBRARY_ROOT="${WORKING_PATH}/library"

ROCBLAS_PATH="${ROCBLAS_ROOT}/rocBLAS-reference"

if [ -n "${ID}" ]; then
    ROCBLAS_PATH="${ROCBLAS_PATH}-${ID}"
fi

TENSILE_LIBRARY_PATH="${LIBRARY_ROOT}/tensile_library/library"
mkdir -p ${TENSILE_LIBRARY_PATH}
TENSILE_LIBRARY_PATH=$(realpath ${TENSILE_LIBRARY_PATH})

BENCHMARK_PATH=${ROCBLAS_PATH}/build/release/clients/staging
ROCBLAS_BENCH=${BENCHMARK_PATH}/rocblas-bench

REFERENCE_PATH=${BENCHMARK_PATH}/results_ref
VERIFICATION_PATH=${BENCHMARK_PATH}/results_validate

mkdir -p ${REFERENCE_PATH}
mkdir -p ${VERIFICATION_PATH}

FILES=$(ls ${SCRIPT_ROOT}/*yaml)

for FILE in $FILES
do
    NAME=$(basename ${FILE} | cut -d'.' -f1)
    ${ROCBLAS_BENCH} --yaml ${FILE} > ${REFERENCE_PATH}/${NAME}.1
done

for FILE in $FILES
do
    NAME=$(basename ${FILE} | cut -d'.' -f1)
    ROCBLAS_TENSILE_LIBPATH=${TENSILE_LIBRARY_PATH} ${ROCBLAS_BENCH} --yaml ${FILE} > ${VERIFICATION_PATH}/${NAME}.1
done


