#!/usr/bin/env bash

HELP_STR="
    Pre-Requisites: >=Anaconda 3.6 (or install python3.6 or higher, python3-pip/pip3, python3-yaml, python3-setuptools, python3-distutils,
                                    python3-venv, wheel, setuptools, pyyaml, matplotlib, pandas, and numpy)
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
    [-h|--help]             Display this help message
    [-o|--output-dir]       Output directory for all tuning-related files
    [-y|--data-type]        Data type of sizes that you want to tune (sgemm, dgemm, hgemm only)
    [-g|--gpu]              GPU used for tuning (arcturus, mi25, mi50, mi60, r7, v340 only)
    [-f|--sclk]             Frequency of sclk in MHz
    [-z|--log]              Pass in log file with rocblas-bench calls, or directory of log files if using network tuning
    [-n|--network]          Optional. String to search for in filenames in log directory
    [-m|--mfma]             Optional. Use MFMA kernels (default=false)
    [-r|--rk]               Optional. Use replacement kernels (sgemm only, default=false)
    [-c|--count]            Optional. Sets all cases where count=1 to count=10 (default=false)
    [-t|--tile-aware]       Optional. Use tile-aware method. (limited support, default=false)
    [-s|--disable-strides]  Optional. Disable leading dimensions and strides in tuning file (default=false)
    [-i|--initialization]   Optional. Initialize matrices when benchmarking (rand_int, trig_float, hpl, default=rand_int)
    [--number]              Optional. Set script number (view scripts/performance in rocBLAS directory, default=1)
    [-u|--username]         Optional. Specify which Tensile fork to use (default=ROCmSoftwarePlatform)
    [--rocblas-fork]        Optional. Specify which rocBLAS fork to use (default=ROCmSoftwarePlatform)
    [-b|--branch]           Optional. Specify which Tensile branch to use (default=master)
    [--rocblas-branch]      Optional. Specify which rocBLAS branch to use (default=develop)
    [-p|--public]           Optional. Specify whether you want to use rocBLAS public repo (default=false)
    [--one-type]            Optional. Only tune one matrix type (nn, nt, or tn)
    [--omit-type]           Optional. Ignore one matrix type when tuning (nn, nt, or tn)
    [--problem-definition]  Optional. Specify gemm, strided batched, or both sizes (gemm, batch, or both, default=both)
    [--hip-clang]           Optional. Use hip-clang compiler (default=false)
    [--rocm-path]           Optional. Define ROCM_PATH, the location of the rocm stack (default=/opt/rocm)
"
HELP=false
COUNT=false
TILE_AWARE=false
MFMA=false
RK=false
DISABLE_STRIDES=false
LIBRARY=vega20
GPU=mi60
DVAL=2
NUM=1
DATA_TYPE=sgemm
PROBLEM_DEFINITION=both
INITIALIZATION=rand_int
ORGANIZATION=ROCmSoftwarePlatform
ROCBLAS_ORGANIZATION=ROCmSoftwarePlatform
ROCBLAS_BRANCH=develop
TENSILE_BRANCH=develop
PUBLIC=false
HIP_CLANG=false
ROCM_PATH=/opt/rocm

OPTS=`getopt -o hg:z:y:o:f:rmctsi:u:b:p --long help,gpu:,log:,network:,data-type:,output-dir:,sclk:,rk,mfma,count,tile-aware,disable-strides,initialization:,username:,branch:,number:,rocblas-fork:,rocblas-branch:,public,one-type:,omit-type:,problem-definition:,hip-clang,rocm-path: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        -h | --help )               HELP=true; shift ;;
        -g | --gpu )                GPU="$2"; shift 2;;
        -z | --log )                LOG="$2"; shift 2;;
        -n | --network )            NETWORK="$2"; shift 2;;
        -y | --data-type )          DATA_TYPE="$2"; shift 2;;
        -o | --output-dir )         OUTPUT_DIR="$2"; shift 2;;
        -f | --sclk )               SCLK="$2"; shift 2;;
        -r | --rk )                 RK=true; shift ;;
        -m | --mfma )               MFMA=true; shift ;;
        -c | --count )              COUNT=true; shift ;;
        -t | --tile-aware )         TILE_AWARE=true; shift ;;
        -s | --disable-strides )    DISABLE_STRIDES=true; shift;;
        -i | --initialization )     INITIALIZATION="$2"; shift 2;;
        -u | --username )           ORGANIZATION="$2"; shift 2;;
        --rocblas-fork )            ROCBLAS_ORGANIZATION="$2"; shift 2;;
        -b | --branch )             TENSILE_BRANCH="$2"; shift 2;;
        --rocblas-branch )          ROCBLAS_BRANCH="$2"; shift 2;;
        -p | --public )             PUBLIC=true; shift ;;
        --number )                  NUM="$2"; shift 2;;
        --one-type )                TUNE_TYPE="$2"; shift 2;;
        --omit-type )               OMIT_TYPE="$2"; shift 2;;
        --problem-definition )      PROBLEM_DEFINITION="$2"; shift 2;;
        --hip-clang )               HIP_CLANG=true; shift ;;
        --rocm-path )               ROCM_PATH="$2"; shift 2;;
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

if [[ "${HIP_CLANG}" == true ]]; then
    TENSILE_COMPILER=hipcc
    ROCBLAS_COMPILER=hip-clang
    export PATH=${ROCM_PATH}/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/llvm/bin:${PATH}
else
    TENSILE_COMPILER=hcc
    ROCBLAS_COMPILER=no-hip-clang
    export PATH=${PATH}:${ROCM_PATH}/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/hcc/bin
fi

collect_uniques () {
    strided=${LOGNAME}-strided.sh
    regular=${LOGNAME}.sh
    strided2=${LOGNAME}2-strided.sh
    regular2=${LOGNAME}2.sh

    pushd scripts
    bash ../../tuning/scripts/unique-rocblas-logs.sh ${strided}
    bash ../../tuning/scripts/unique-rocblas-logs.sh ${regular}
    mv unique-${strided} ${strided}
    mv unique-${regular} ${regular}
    popd

    pushd scripts2
    bash ../../tuning/scripts/unique-rocblas-logs.sh ${strided2}
    bash ../../tuning/scripts/unique-rocblas-logs.sh ${regular2}
    mv unique-${strided2} ${strided2}
    mv unique-${regular2} ${regular2}
    popd
}

run_tune_nn () {
    NN=build-${LIBRARY}-${DATA_TYPE}-nn-${OUTPUT_DIR}
    mkdir ${NN}
    cp ../configs/*nn*.yaml ${NN}

    pushd ${NN}
    echo "#!/bin/sh" > tune.sh
    echo "touch time.begin" >> tune.sh
    echo "../Tensile/bin/Tensile ${LIBRARY}_${DATA_TYPE}_nn_${OUTPUT_DIR}.yaml ./ --cxx-compiler=${TENSILE_COMPILER} 2>&1 | tee make.out" >> tune.sh
    echo "touch time.end" >> tune.sh
    chmod 755 tune.sh
    ./tune.sh
    popd

    cp ${NN}/3_LibraryLogic/* exact/
}

run_tune_nt () {
    NT=build-${LIBRARY}-${DATA_TYPE}-nt-${OUTPUT_DIR}
    mkdir ${NT}
    cp ../configs/*nt*.yaml ${NT}

    pushd ${NT}
    echo "#!/bin/sh" > tune.sh
    echo "touch time.begin" >> tune.sh
    echo "../Tensile/bin/Tensile ${LIBRARY}_${DATA_TYPE}_nt_${OUTPUT_DIR}.yaml ./ --cxx-compiler=${TENSILE_COMPILER} 2>&1 | tee make.out" >> tune.sh
    echo "touch time.end" >> tune.sh
    chmod 755 tune.sh
    ./tune.sh
    popd

    cp ${NT}/3_LibraryLogic/* exact/
}

run_tune_tn () {
    TN=build-${LIBRARY}-${DATA_TYPE}-tn-${OUTPUT_DIR}
    mkdir ${TN}
    cp ../configs/*tn*.yaml ${TN}

    pushd ${TN}
    echo "#!/bin/sh" > tune.sh
    echo "touch time.begin" >> tune.sh
    echo "../Tensile/bin/Tensile ${LIBRARY}_${DATA_TYPE}_tn_${OUTPUT_DIR}.yaml ./ --cxx-compiler=${TENSILE_COMPILER} 2>&1 | tee make.out" >> tune.sh
    echo "touch time.end" >> tune.sh
    chmod 755 tune.sh
    ./tune.sh
    popd

    cp ${TN}/3_LibraryLogic/* exact/
}

run_tune_all_scripts () {
    mkdir exact
    if [[ "${TUNE_TYPE}" == nn ]]; then
        run_tune_nn
    elif [[ "${TUNE_TYPE}" == nt ]]; then
        run_tune_nt
    elif [[ "${TUNE_TYPE}" == tn ]]; then
        run_tune_tn
    elif [[ "${OMIT_TYPE}" == nn ]]; then
        run_tune_nt
        run_tune_tn
    elif [[ "${OMIT_TYPE}" == nn ]]; then
        run_tune_nt
        run_tune_tn
    elif [[ "${OMIT_TYPE}" == nt ]]; then
        run_tune_nn
        run_tune_tn
    elif [[ "${OMIT_TYPE}" == tn ]]; then
        run_tune_nn
        run_tune_nt
    else
        run_tune_nn
        run_tune_nt
        run_tune_tn
    fi
}

make_packages()
{
    make package
    make package_clients
    cp *.deb ../../../packages/library
    cp clients/*.deb ../../../packages/client
}

mkdir ${OUTPUT_DIR}
EXTRACT_SIZE_PATH=`pwd`/${OUTPUT_DIR}
if [ -z ${NETWORK+foo} ]; then
    python tuning/automation/GenerateTuningConfigurations.py ${LOG} ${EXTRACT_SIZE_PATH} ${OUTPUT_DIR}.yaml ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION} ${INITIALIZATION}
else
    python tuning/automation/GenerateTuningConfigurations.py ${LOG} ${NETWORK} ${EXTRACT_SIZE_PATH} ${OUTPUT_DIR}.yaml ${LIBRARY} ${TILE_AWARE} ${MFMA} ${RK} ${DISABLE_STRIDES} ${PROBLEM_DEFINITION} ${INITIALIZATION}
fi

pushd ${OUTPUT_DIR}
LOGNAME="${LOG%.*}"
collect_uniques
chmod 755 scripts/*
chmod 755 scripts2/*
git clone https://github.com/${ORGANIZATION}/Tensile.git -b ${TENSILE_BRANCH}

pushd Tensile
run_tune_all_scripts
popd

REPO=rocBLAS-internal
if [[ "${PUBLIC}" == true ]]; then
    REPO=rocBLAS
fi

git clone https://github.com/${ROCBLAS_ORGANIZATION}/${REPO}.git -b ${ROCBLAS_BRANCH} rocBLAS
mkdir library
mv Tensile/exact library/
mkdir library/merge

DIR=archive
if [[ "${LIBRARY}" == arcturus ]]; then
    DIR=asm_full
fi
python Tensile/Tensile/Utilities/merge_rocblas_yaml_files.py rocBLAS/library/src/blas3/Tensile/Logic/${DIR} library/exact library/merge

if [[ "${LIBRARY}" != arcturus ]]; then
    python rocBLAS/library/src/blas3/Tensile/Logic/archive/massage.py library/merge library/massage
fi

mkdir packages
mkdir packages/library
mkdir packages/client
pushd rocBLAS
./install.sh -c --build_dir reference-build --${ROCBLAS_COMPILER} 2>&1 | tee log-reference-build
pushd reference-build/release
popd

if [[ "${LIBRARY}" != arcturus ]]; then
    cp ../library/massage/* library/src/blas3/Tensile/Logic/asm_full
    cp ../library/massage/* library/src/blas3/Tensile/Logic/asm_ci
    cp ../library/merge/* library/src/blas3/Tensile/Logic/archive
else
    cp ../library/merge/* library/src/blas3/Tensile/Logic/asm_full
fi
./install.sh -c --build_dir tuned-build --${ROCBLAS_COMPILER} 2>&1 | tee log-tuned-build
pushd tuned-build/release
make_packages
popd

cp ../scripts/*.sh reference-build/release/clients/staging
cp ../scripts/*.sh tuned-build/release/clients/staging
pushd reference-build/release/clients/staging
./doit_all1.sh
find results1 -name \*.1 -exec sed -i "s/4t/t/g" {} \;
find results1 -name \*.1 -exec sed -i "s/4r/r/g" {} \;
./*verify.sh 2>&1 | tee log-verification-build
popd

pushd tuned-build/release/clients/staging
./doit_all1.sh
find results1 -name \*.1 -exec sed -i "s/4t/t/g" {} \;
find results1 -name \*.1 -exec sed -i "s/4r/r/g" {} \;
./*verify.sh 2>&1 | tee log-verification-tuned-build
popd

mv ../scripts/*-all.sh scripts/performance/${OUTPUT_DIR}${NUM}.sh
popd
popd

source ~/.bashrc
./tuning/scripts/analyze-results.sh -o ${OUTPUT_DIR}/analysis -r ${OUTPUT_DIR}/rocBLAS/reference-build/release/clients/staging/results1 -b ${OUTPUT_DIR}/rocBLAS/tuned-build/release/clients/staging/results1 -z ${LOG} -f ${SCLK} -s ${DVAL} -g ${GPU} -c ${COUNT} -m ${MFMA}
