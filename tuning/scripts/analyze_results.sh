#!/bin/sh

HELP=false
PLOT=true
MFMA=false
COUNT=false

HELP_STR="
Usage: ${0} WORKING_PATH LOG_PATH LIBRARY [options]

  where LIBRARY = {arcturus | vega20 | vega10 | mi25 | r9nano | hip}

Options:
-h | --help             Display this help message
-s | --size             Data size
-f | --sclk             Frequency of sclk in MHz
-m | --mfma             Was MFMA enabled
-c | --count            Sets all cases where count=1 to count=10
-n | --no-plot          Skip plotting
"

if ! OPTS=$(getopt -o h,s:,f:,m,c,n \
--long help,size:,sclk:,mfma,count,no-plot -n 'parse-options' -- "$@")
then
  echo "Failed parsing options"
  exit 1
fi

eval set -- "${OPTS}"

while true; do
  case ${1} in
    -h | --help )              HELP=true; shift ;;
    -s | --size)               SZ=${2}; shift 2;;
    -f | --sclk)               SCLK=${2}; shift 2;;
    -m | --mfma )	             MFMA=true; shift ;;
    -c | --count )	           COUNT=true; shift ;;
    -n | --no-plot )           PLOT=false; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ${HELP}; then
  echo "${HELP_STR}"
  exit 0
fi

if [ -z ${SZ+x} ]; then
  echo "Data size not specified: assuming 4 bytes"
  SZ=4
fi

if [ -z ${SCLK+x} ]; then
  echo "Clock rate not specified: assuming 1000 MHz"
  SCLK=1000
fi

if [ $# != 3 ]; then
  echo "Exactly three possitional args required"
  echo "See ${0} --help"
fi

WORKING_PATH=${1}
LOG=${2}
LIBRARY=${3}

REFERENCE_PATH=${WORKING_PATH}/benchmarks/reference
TUNED_PATH=${WORKING_PATH}/benchmarks/tuned

ANALYSIS_REF_PATH=${WORKING_PATH}/analysis/reference
ANALYSIS_TUNED_PATH=${WORKING_PATH}/analysis/tuned
ANALYSIS_FINAL_PATH=${WORKING_PATH}/analysis/comparison

mkdir -p "${ANALYSIS_REF_PATH}"
mkdir -p "${ANALYSIS_TUNED_PATH}"
mkdir -p "${ANALYSIS_FINAL_PATH}"

# determine full path of tools root
TOOLS_ROOT=$(realpath "${0}" | xargs dirname | xargs dirname)
AUTOMATION_ROOT=${TOOLS_ROOT}/automation

ANALYSIS=${AUTOMATION_ROOT}/PerformanceAnalysis.py
COMPARE=${AUTOMATION_ROOT}/CompareResults.py
PLOT_RESULTS=${AUTOMATION_ROOT}/PlotResults.py

# run analysis of reference and tuned benchmarks
python3 "${ANALYSIS}" "${REFERENCE_PATH}" "${ANALYSIS_REF_PATH}" "${SCLK}" \
  "${SZ}" "${LOG}" "${LIBRARY}" "${MFMA}" "${COUNT}"
python3 "${ANALYSIS}" "${TUNED_PATH}" "${ANALYSIS_TUNED_PATH}" "${SCLK}" \
  "${SZ}" "${LOG}" "${LIBRARY}" "${MFMA}" "${COUNT}"

# compare reference and tuned
find "${ANALYSIS_TUNED_PATH}" -name '*aggregated*' -print0 \
  | xargs -0 -n1 basename \
  | xargs -I{} \
  python3 "${COMPARE}" "${ANALYSIS_REF_PATH}/{}" "${ANALYSIS_TUNED_PATH}/{}" "${ANALYSIS_FINAL_PATH}/{}"

if $PLOT; then

  REFERENCE_PLOT=${ANALYSIS_REF_PATH}/plot
  TUNED_PLOT=${ANALYSIS_TUNED_PATH}/plot

  mkdir -p "${REFERENCE_PLOT}"
  mkdir -p "${TUNED_PLOT}"

  # plot reference
  AGGREGATED_FILES=$(ls "${ANALYSIS_REF_PATH}"/*aggregated*)
  for FILE in ${AGGREGATED_FILES}; do
    FILENAME=$(basename "${FILE}")
    NAMEPART="${FILENAME%-aggregated.*}"

    python3 "${PLOT_RESULTS}" "${FILE}" "${REFERENCE_PLOT}/${NAMEPART}"
  done

  # plot tuned
  AGGREGATED_FILES=$(ls "${ANALYSIS_TUNED_PATH}"/*aggregated*)
  for FILE in ${AGGREGATED_FILES}; do
    FILENAME=$(basename "$FILE")
    NAMEPART="${FILENAME%-aggregated.*}"

    python3 "${PLOT_RESULTS}" "${FILE}" "${TUNED_PLOT}/${NAMEPART}"
  done

fi
