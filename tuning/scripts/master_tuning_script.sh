#!/usr/bin/env bash

HELP_STR="
    Pre-Requisites: >=Anaconda 3.6 (or install python3.6 or higher, python3-pip/pip3, python3-yaml, python3-setuptools, python3-distutils python3-venv, wheel, setuptools, pyyaml, matplotlib, pandas, and numpy)
                    >=llvm-6.0-dev, >=cmake3.5, zlib1g-dev
                    <=rocm3.0 stack for hcc, >=rocm3.3 stack for hip-clang
    About
    This is the master tuning script. Here's what it does: 
    1) Generates an input yaml file (ROCBLAS_LAYER=2 and ROCBLAS_LAYER=4 are supported) for each of the NN/NT/TN configurations.
    2) Runs Tensile for all 3 input yaml files
    3) Merges the tuning results (yaml file in 3_LibraryLogic directory) into the existing rocBLAS
    4) Runs massage script (adds ldc=ldd sizes) for vega10 and vega20 only
    5) Builds rocBLAS without tuned sizes (build_dir=reference_build)
    6) Builds rocBLAS with tuned sizes (build_dir=tuned_build)
    7) Runs rocblas-bench with untuned sizes and tuned sizes
    8) Runs analysis script, providing spreadsheets with performance before and after tuning

    usage: $0 
    [-h|--help]         Display this help message
    [-o|--output-dir]   Output directory for all tuning-related files
    [-y|--data-type]    Data type of sizes that you want to tune (sgemm, dgemm, hgemm only)
    [-g|--gpu]          GPU used for tuning (arcturus, mi25, mi50, mi60, r7, v340 only)
    [-f|--sclk]         Frequency of sclk in MHz 
    [-z|--log]          Pass in log file with rocblas-bench calls, or directory of log files if using network tuning
    [-n|--network]      Optional. String to search for in filenames in log directory
    [-m|--mfma]         Optional. Use MFMA kernels (default=false)
    [-r|--rk]           Optional. Use replacement kernels (sgemm only, default=false)
    [-c|--count]        Optional. Sets all cases where count=1 to count=10 (default=false)
    [-t|--tile-aware]   Optional. Use tile-aware method. (limited support, default=false) 
    [--number]          Optional. Set script number (view scripts/performance in rocBLAS directory, default=1)
    [-u|--username]     Optional. Specify which rocBLAS/Tensile fork to use (default=ROCmSoftwarePlatform)
    [-b|--branch]       Optional. Specify which rocBLAS/Tensile branch to use (default=develop)
"
HELP=false
COUNT=false
TILE_AWARE=false
MFMA=false
RK=false
LIBRARY=vega20
GPU=mi60
DVAL=2
NUM=1
DATA_TYPE=sgemm
ORGANIZATION=ROCmSoftwarePlatform
BRANCH=develop

OPTS=`getopt -o hg:z:y:o:f:rmctu:b: --long help,gpu:,log:,network:,data-type:,output_dir:,sclk:,rk,mfma,count,tile-aware,username:,branch:,number: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        -h | --help )         HELP=true; shift ;;
        -g | --gpu )          GPU="$2"; shift 2;;
        -z | --log )          LOG="$2"; shift 2;;
        -n | --network )      NETWORK="$2"; shift 2;;
        -y | --data-type )    DATA_TYPE="$2"; shift 2;;
        -o | --output-dir )   OUTPUT_DIR="$2"; shift 2;;
        -f | --sclk )         SCLK="$2"; shift 2;;
        -r | --rk )           RK=true; shift ;;
        -m | --mfma )         MFMA=true; shift ;;
        -c | --count )        COUNT=true; shift ;;
        -t | --tile-aware )   TILE_AWARE=true; shift ;;
        -u | --username )     ORGANIZATION="$2"; shift 2;;
        -b | --branch )       BRANCH="$2"; shift 2;;
        --number )            NUM="$2"; shift 2;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

if $HELP; then
    echo "${HELP_STR}" >&2
    exit 2
fi

if [ -z ${LOG+foo} ]; then
   printf "A problem specification file or directory is required\n"
   exit 2
fi

if [ -z ${OUTPUT_DIR+foo} ]; then
   printf "An output directory is required\n"
   exit 2
fi

if [[ "${DATA_TYPE}" == hgemm || "${DATA_TYPE}" == h ]]; then
    DATA_TYPE=hgemm
    DVAL=4
elif [[ "${DATA_TYPE}" == dgemm || "${DATA_TYPE}" == d ]]; then
    DATA_TYPE=dgemm
    DVAL=1
else
    printf "Assuming sgemm\n"
    DATA_TYPE=sgemm
    DVAL=2
fi

if [[ "${GPU}" == mi25 || "${GPU}" == v340 ]]; then
    LIBRARY=vega10
elif [[ "${GPU}" == arcturus ]]; then
    LIBRARY=arcturus
elif [[ "${GPU}" == mi50 || "${GPU}" == r7 ]]; then
    GPU=mi50
else
    printf "Assuming vega20 gpu library\n"
    GPU=mi60
fi

run_tune_all_scripts () {
    NN=build-${LIBRARY}-${DATA_TYPE}-nn-${OUTPUT_DIR}
    NT=build-${LIBRARY}-${DATA_TYPE}-nt-${OUTPUT_DIR}
    TN=build-${LIBRARY}-${DATA_TYPE}-tn-${OUTPUT_DIR}

    mkdir ${NN}
    mkdir ${NT}
    mkdir ${TN}
    cp ../configs/*nn*.yaml ${NN}
    cp ../configs/*nt*.yaml ${NT}
    cp ../configs/*tn*.yaml ${TN}

    echo "#!/bin/sh" > tune-all.sh
    echo "for dir in ${NN} ${NT} ${TN}" >> tune-all.sh
    echo "do" >> tune-all.sh
    echo "  cd \${dir}" >> tune-all.sh
    echo "  ./tune.sh > tune-errs 2>&1" >> tune-all.sh
    echo "  cd .." >> tune-all.sh
    echo "done" >> tune-all.sh
    chmod 755 tune-all.sh

    pushd ${NN}
    echo "#!/bin/sh" > tune.sh
    echo "touch time.begin" >> tune.sh
    echo "../Tensile/bin/Tensile ${LIBRARY}_${DATA_TYPE}_nn_${OUTPUT_DIR}.yaml ./ 2>&1 | tee make.out" >> tune.sh
    echo "touch time.end" >> tune.sh
    chmod 755 tune.sh
    popd
    
    pushd ${NT}
    echo "#!/bin/sh" > tune.sh
    echo "touch time.begin" >> tune.sh
    echo "../Tensile/bin/Tensile ${LIBRARY}_${DATA_TYPE}_nt_${OUTPUT_DIR}.yaml ./ 2>&1 | tee make.out" >> tune.sh
    echo "touch time.end" >> tune.sh
    chmod 755 tune.sh
    popd

    pushd ${TN}
    echo "#!/bin/sh" > tune.sh
    echo "touch time.begin" >> tune.sh
    echo "../Tensile/bin/Tensile ${LIBRARY}_${DATA_TYPE}_tn_${OUTPUT_DIR}.yaml ./ 2>&1 | tee make.out" >> tune.sh
    echo "touch time.end" >> tune.sh
    chmod 755 tune.sh
    popd

    ./tune-all.sh
    
    mkdir exact
    cp ${NN}/3_LibraryLogic/* exact/
    cp ${NT}/3_LibraryLogic/* exact/
    cp ${TN}/3_LibraryLogic/* exact/
}

mkdir ${OUTPUT_DIR}
EXTRACT_SIZE_PATH=`pwd`/${OUTPUT_DIR}
if [ -z ${NETWORK+foo} ]; then
    python tuning/automation/GenerateTuningConfigurations.py ${LOG} ${EXTRACT_SIZE_PATH} ${OUTPUT_DIR}.yaml ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK}
else
    python tuning/automation/GenerateTuningConfigurations.py ${LOG} ${NETWORK} ${EXTRACT_SIZE_PATH} ${OUTPUT_DIR}.yaml ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK}
fi

pushd ${OUTPUT_DIR}
chmod 755 scripts/*
chmod 755 scripts2/*
git clone https://github.com/${ORGANIZATION}/Tensile.git -b ${BRANCH}

pushd Tensile
run_tune_all_scripts
popd

git clone https://github.com/${ORGANIZATION}/rocBLAS-internal.git -b ${BRANCH} rocBLAS
mkdir library
mv Tensile/exact library/
mkdir library/merge
python Tensile/Tensile/Utilities/merge_rocblas_yaml_files.py rocBLAS/library/src/blas3/Tensile/Logic/archive library/exact library/merge

if [[ "${LIBRARY}" != arcturus ]]; then
    python rocBLAS/library/src/blas3/Tensile/Logic/archive/massage.py library/merge library/massage
fi

pushd rocBLAS
./install.sh -c --build_dir reference-build 2>&1 | tee log-reference-build

if [[ "${LIBRARY}" != arcturus ]]; then
    cp ../library/massage/* library/src/blas3/Tensile/Logic/asm_full
    cp ../library/massage/* library/src/blas3/Tensile/Logic/asm_ci
    cp ../library/merge/* library/src/blas3/Tensile/Logic/archive
else
    cp ../library/merge/* library/src/blas3/Tensile/Logic/asm_full
fi
./install.sh -c --build_dir tuned-build 2>&1 | tee log-tuned-build

cp ../scripts/*.sh reference-build/release/clients/staging
cp ../scripts/*.sh tuned-build/release/clients/staging
pushd reference-build/release/clients/staging
./doit_all1.sh
find results1 -name \*.1 -exec sed -i "s/4t/t/g" {} \;
popd

pushd tuned-build/release/clients/staging
./doit_all1.sh
find results1 -name \*.1 -exec sed -i "s/4t/t/g" {} \;
popd

mv ../scripts/${OUTPUT_DIR}-all.sh scripts/performance/${OUTPUT_DIR}${NUM}.sh
popd
popd

source ~/.bashrc
./tuning/scripts/analyze-results.sh -o ${OUTPUT_DIR}/analysis -r ${OUTPUT_DIR}/rocBLAS/reference-build/release/clients/staging/results1 -b ${OUTPUT_DIR}/rocBLAS/tuned-build/release/clients/staging/results1 -z ${LOG} -f ${SCLK} -s ${DVAL} -g ${GPU} -c ${COUNT} -m ${MFMA}
