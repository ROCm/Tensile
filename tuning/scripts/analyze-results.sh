#!/bin/sh

HELP_STR="usage: $0 [-b|--benchmark-path <benchmark results path>] [-r| --reference-path <reference results path>] [-o|--output <output path>] [-f] [-s] [-z] [-h|--help]"
HELP=false

OPTS=`getopt -o hf:s:b:o:r:z: --long help,output-path:,reference-path:,benchmark-path: -n '
parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
  case "$1" in
    -h | --help )              HELP=true; shift ;;
    -r | --reference-path )    REFERENCE_PATH="$2"; shift 2;;
    -b | --benchmark-path  )   BENCHMARK_PATH="$2"; shift 2;;
    -o | --output-path )       OUTPUT_PATH="$2"; shift 2;;
    -z )                       LOG="$2"; shift 2;;
    -f )                       FREQ="$2"; shift 2;;
    -s )                       SZ="$2"; shift 2;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if $HELP; then
  echo "${HELP_STR}" >&2
  exit 2
fi

if [ -z ${REFERENCE_PATH+foo} ]; then
   printf "Need a reference results path\n"
   exit 2
fi

if [ -z ${BENCHMARK_PATH+foo} ]; then
   printf "Need the benchmark results path\n"
   exit 2
fi

if [ -z ${OUTPUT_PATH+foo} ]; then
   printf "Need the output path\n"
   exit 2
fi

if [ -z ${LOG+foo} ]; then
   printf "Need to select a log file\n"
   exit 2
fi

if [ -z ${FREQ+foo} ]; then
   printf "Need clock rate\n"
   exit 2
fi

if [ -z ${SZ+foo} ]; then
   printf "Need data size\n"
   exit 2
fi

CASE_REFERENCE=${OUTPUT_PATH}/reference
CASE_NEW=${OUTPUT_PATH}/new
CASE_FINAL=${OUTPUT_PATH}/final

mkdir -p ${CASE_REFERENCE}
mkdir -p ${CASE_NEW}
mkdir -p ${CASE_FINAL}

REFERENCE_RESULTS=${CASE_REFERENCE}/results
REFERENCE_AGGREGATED=${CASE_REFERENCE}/aggregated
NEW_RESULTS=${CASE_NEW}/results
NEW_AGGREGATED=${CASE_NEW}/aggregated

mkdir -p ${REFERENCE_RESULTS}
mkdir -p ${REFERENCE_AGGREGATED}
mkdir -p ${NEW_RESULTS}
mkdir -p ${NEW_AGGREGATED}

cp ${REFERENCE_PATH}/* ${REFERENCE_RESULTS}
cp ${BENCHMARK_PATH}/* ${NEW_RESULTS}

#determing full path of tools root
TOOLS_ROOT=`dirname "$0"`
TOOLS_ROOT=`( cd "${TOOLS_ROOT}" && cd .. && pwd )`
AUTOMATION_ROOT="${TOOLS_ROOT}/automation"

ANALYSIS=${AUTOMATION_ROOT}/PerformanceAnalysis.py
COMPARE=${AUTOMATION_ROOT}/CompareResults.py
PLOT_DIFF=${AUTOMATION_ROOT}/PlotDifference.py
PLOT_RESULTS=${AUTOMATION_ROOT}/PlotResults.py


python ${ANALYSIS} ${LOG} ${REFERENCE_RESULTS} ${REFERENCE_AGGREGATED} ${FREQ} ${SZ}
python ${ANALYSIS} ${LOG} ${NEW_RESULTS} ${NEW_AGGREGATED} ${FREQ} ${SZ}


ls ${NEW_AGGREGATED}/*aggregated* | xargs -n1 basename | xargs -I{} python ${COMPARE} ${REFERENCE_AGGREGATED}/{} ${NEW_AGGREGATED}/{} ${CASE_FINAL}/{}

REFERENCE_PLOT=${CASE_REFERENCE}/plot
NEW_PLOT=${CASE_NEW}/plot

mkdir -p ${REFERENCE_PLOT}
mkdir -p ${NEW_PLOT}



aggregated_files=$(ls ${REFERENCE_AGGREGATED}/*aggregated*)
for file in ${aggregated_files}; do
  filename=$(basename "$file")
  namepart="${filename%-aggregated.*}"

  python ${PLOT_RESULTS} ${file} ${REFERENCE_PLOT}/${namepart}
done

aggregated_files=$(ls ${NEW_AGGREGATED}/*aggregated*)
for file in ${aggregated_files}; do
  filename=$(basename "$file")
  namepart="${filename%-aggregated.*}"

  python ${PLOT_RESULTS} ${file} ${NEW_PLOT}/${namepart}
done




