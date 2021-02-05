#!/bin/bash

HELP=false

HELP_STR="
Usage: ${0} WORKING_PATH ROCBLAS_PATH

Options:
-h | --help             Display this help message
"

if ! OPTS=$(getopt -o h --long help -n 'parse-options' -- "$@")
then
  echo "Failed parsing options"
  exit 1
fi

eval set -- "${OPTS}"

while true; do
  case ${1} in
    -h | --help )         HELP=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ${HELP}; then
  echo "${HELP_STR}"
  exit 0
fi

if [ $# != 2 ]; then
  echo "Exactly two positional args required"
  echo "See ${0} --help"
  exit 2
fi

WORKING_PATH=${1}
ROCBLAS_PATH=${2}

SCRIPT_ROOT=${WORKING_PATH}/scripts
LIBRARY_ROOT=${WORKING_PATH}/library

TENSILE_LIBRARY_PATH=${LIBRARY_ROOT}/tensile_library/library
ROCBLAS_BENCH=${ROCBLAS_PATH}/build/release/clients/staging/rocblas-bench

REFERENCE_PATH=${WORKING_PATH}/benchmarks/reference
TUNED_PATH=${WORKING_PATH}/benchmarks/tuned

mkdir -p "${REFERENCE_PATH}"
mkdir -p "${TUNED_PATH}"

FILES=$(ls "${SCRIPT_ROOT}"/*yaml)
echo ${FILES}
echo "Benchmarking reference library"
for FILE in $FILES
do
  NAME=$(basename "${FILE}" | cut -d'.' -f1)
  "${ROCBLAS_BENCH}" --yaml "${FILE}" > "${REFERENCE_PATH}/${NAME}.1"
done

echo "Benchmarking tuned library"
for FILE in $FILES
do
  NAME=$(basename "${FILE}" | cut -d'.' -f1)
  ROCBLAS_TENSILE_LIBPATH="${TENSILE_LIBRARY_PATH}" "${ROCBLAS_BENCH}" --yaml "${FILE}" > "${TUNED_PATH}/${NAME}.1"
done
