#!/usr/bin/env bash

HELP_STR="
    Pre-Requisites: >=Anaconda 3.6 (or install python3.6 or higher, python3-pip/pip3, python3-yaml, python3-setuptools, python3-distutils,
                                    python3-venv, wheel, setuptools, pyyaml, msgpack, matplotlib, pandas, and numpy)
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
    [-z|--log]              Pass in log file with rocblas-bench calls, or directory of log files if using network tuning
    [-y|--data-type]        Optional. Data type of sizes that you want to tune (sgemm, dgemm, hgemm only)
    [-g|--gpu]              Optional. GPU used for tuning (arcturus, mi25, mi50, mi60, r7, v340 only)
    [-f|--sclk]             Optional. Frequency of sclk in MHz 
    [-d|--dependencies]     Optional. Install required dependencies (dependencies are not installed by default)
    [-n|--network]          Optional. String to search for in filenames in log directory
    [--client]              Optional. Choose Tensile client version. (new, old, both, default=new)
    [-m|--mfma]             Optional. Use MFMA kernels (default=false)
    [-r|--rk]               Optional. Use replacement kernels (sgemm only, default=false)
    [-c|--count]            Optional. Sets all cases where count=1 to count=10 (default=false)
    [-t|--tile-aware]       Optional. Use tile-aware method. (limited support, default=false)
    [-s|--disable-strides]  Optional. Disable leading dimensions and strides in tuning file (default=false)
    [-v|--verification]     Optional. Enable verification when running sizes in rocblas-bench (default=false, always true if arcturus)
    [-i|--initialization]   Optional. Initialize matrices when benchmarking (rand_int, trig_float, hpl, default=rand_int)
    [--disable-hpa]         Optional. Disable high precision accumulate in hgemm sizes (default=false)
    [--number]              Optional. Set script number (view scripts/performance in rocBLAS directory, default=1)
    [-u|--username]         Optional. Specify which Tensile fork to use (default=ROCmSoftwarePlatform)
    [--rocblas-fork]        Optional. Specify which rocBLAS fork to use (default=ROCmSoftwarePlatform)
    [-b|--branch]           Optional. Specify which Tensile branch to use (default=develop)
    [--rocblas-branch]      Optional. Specify which rocBLAS branch to use (default=develop)
    [-p|--public]           Optional. Specify whether you want to use rocBLAS (public) repo (default=false)
    [--one-type]            Optional. Only tune one matrix type (nn, nt, or tn)
    [--omit-type]           Optional. Ignore one matrix type when tuning (nn, nt, or tn)
    [--problem-definition]  Optional. Specify gemm, strided batched, or both sizes (gemm, batch, or both, default=both)
    [--hcc]                 Optional. Use hcc compiler (default=false)
    [--rocm-path]           Optional. Define ROCM_PATH, the location of the rocm stack (default=/opt/rocm)
"
HELP=false
TENSILE_CLIENT=new
COUNT=false
TILE_AWARE=false
MFMA=false
RK=false
DISABLE_STRIDES=false
DISABLE_HPA=false
LIBRARY=vega20
GPU=mi60
DVAL=2
NUM=1
DATA_TYPE=sgemm
PROBLEM_DEFINITION=both
VERIFICATION=false
INITIALIZATION=rand_int
ORGANIZATION=ROCmSoftwarePlatform
ROCBLAS_ORGANIZATION=ROCmSoftwarePlatform
ROCBLAS_BRANCH=develop
TENSILE_BRANCH=develop
PUBLIC=false
HCC=false
ROCM_PATH=/opt/rocm
DEPENDENCIES=false

OPTS=`getopt -o hg:z:y:o:f:drmn:ctsvi:u:b:p --long help,gpu:,log:,network:,data-type:,output-dir:,sclk:,dependencies,client:,rk,mfma,count,tile-aware,disable-strides,verification,initialization:,disable-hpa,username:,branch:,number:,rocblas-fork:,rocblas-branch:,public,one-type:,omit-type:,problem-definition:,hcc,rocm-path: -n 'parse-options' -- "$@"`

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
        -d | --dependencies )       DEPENDENCIES=true; shift ;;
        -r | --rk )                 RK=true; shift ;;
        -m | --mfma )               MFMA=true; shift ;;
        -c | --count )              COUNT=true; shift ;;
        -t | --tile-aware )         TILE_AWARE=true; shift ;;
        -s | --disable-strides )    DISABLE_STRIDES=true; shift ;;
        -v | --verification )       VERIFICATION=true; shift ;;
        -i | --initialization )     INITIALIZATION="$2"; shift 2;;
        --disable-hpa )             DISABLE_HPA=true; shift;;
        -u | --username )           ORGANIZATION="$2"; shift 2;;
        --rocblas-fork )            ROCBLAS_ORGANIZATION="$2"; shift 2;;
        -b | --branch )             TENSILE_BRANCH="$2"; shift 2;;
        --rocblas-branch )          ROCBLAS_BRANCH="$2"; shift 2;;
        -p | --public )             PUBLIC=true; shift ;;
        --client)                   TENSILE_CLIENT="$2"; shift 2;;
        --number )                  NUM="$2"; shift 2;;
        --one-type )                TUNE_TYPE="$2"; shift 2;;
        --omit-type )               OMIT_TYPE="$2"; shift 2;;
        --problem-definition )      PROBLEM_DEFINITION="$2"; shift 2;;
        --hcc )                     HCC=true; shift ;;
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

if [[ "${DEPENDENCIES}" == true ]]; then
    # install dependencies for Tensile and rocBLAS (Ubuntu only)
    sudo apt install -y --no-install-recommends cmake make ca-certificates git \
    pkg-config python3 python3-dev python3-matplotlib python3-pandas python3-pip \
    python3-setuptools python3-tk python3-venv python3-yaml libnuma1 llvm-6.0-dev \
    libboost-all-dev zlib1g-dev libomp-dev gfortran libpthread-stubs0-dev libmsgpack-dev \
    libmsgpackc2 wget

    # add required python dependencies
    pip3 install setuptools --upgrade && pip3 install wheel && pip3 install pyyaml msgpack

    # download and install Anaconda to ensure the spreadsheet can be generated
    if [[ $(ls -A ~/anaconda | wc -c) -eq 0 ]]; then
        wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda3.7.sh && \
        bash ~/anaconda3.7.sh -b -p ~/anaconda && eval "$(~/anaconda/bin/conda shell.bash hook)"
    fi

    # Install Gtest
    if [[ $(ls -A /usr/src/gtest/googletest-release-1.10.0 | wc -c) -eq 0 ]]; then
        sudo mkdir -p /usr/src/gtest && pushd /usr/src/gtest && \
        sudo wget https://github.com/google/googletest/archive/release-1.10.0.tar.gz  && \
        sudo tar -xvf release-1.10.0.tar.gz  && \
        pushd googletest-release-1.10.0 && \
        sudo mkdir build && pushd build && sudo cmake .. && sudo make && sudo make install && popd && popd && popd
    fi

    # Install Lapack
    if [[ $(ls -A /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1 | wc -c) -eq 0 ]]; then
        sudo mkdir -p /usr/src/lapack && pushd /usr/src/lapack && \
        sudo wget https://github.com/Reference-LAPACK/lapack-release/archive/lapack-3.7.1.tar.gz  && \
        sudo tar -xvf lapack-3.7.1.tar.gz  && \
        pushd lapack-release-lapack-3.7.1 && \
        sudo mkdir build && pushd build && \
        sudo cmake .. -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF -DCMAKE_Fortran_FLAGS='-fno-optimize-sibling-calls' && \
        sudo make && sudo make install && popd && popd && popd
    fi
fi

if [[ "${HCC}" == true ]]; then
    TENSILE_COMPILER=hcc
    CODE_OBJECT_VERSION=V2
    ROCBLAS_COMPILER=no-hip-clang
    export PATH=${ROCM_PATH}/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/hcc/bin:${PATH}
else
    TENSILE_COMPILER=hipcc
    CODE_OBJECT_VERSION=V3
    ROCBLAS_COMPILER=hip-clang
    export PATH=${PATH}:${ROCM_PATH}/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/llvm/bin
fi

if [[ "${DATA_TYPE}" == hgemm || "${DATA_TYPE}" == h ]]; then
    DATA_TYPE=hgemm
    DVAL=4
elif [[ "${DATA_TYPE}" == dgemm || "${DATA_TYPE}" == d ]]; then
    DATA_TYPE=dgemm
    DVAL=1
else
    if [[ "$(grep -c 'r s' ${LOG})" -gt 0 || "$(grep -c 'r f32' ${LOG})" -gt 0 || "$(grep -c 'sgemm' ${LOG})" -gt 0 || "$(grep -c 'a_type f32' ${LOG})" -gt 0 || "$(grep -c '"a_type: "f32_r"' ${LOG})" -gt 0 ]]; then
        printf "sgemm detected\n"
        DATA_TYPE=sgemm
        DVAL=2
    elif [[ "$(grep -c 'r d' ${LOG})" -gt 0 || "$(grep -c 'r f64' ${LOG})" -gt 0 || "$(grep -c 'dgemm' ${LOG})" -gt 0 ]]; then
        printf "dgemm detected\n"
        DATA_TYPE=dgemm
        DVAL=1
    elif [[ "$(grep -c 'r h' ${LOG})" -gt 0 || "$(grep -c 'r f16' ${LOG})" -gt 0 || "$(grep -c 'hgemm' ${LOG})" -gt 0 || "$(grep -c 'a_type f16' ${LOG})" -gt 0 || "$(grep -c 'a_type: "f16_r"' ${LOG})" -gt 0 ]]; then
        printf "hgemm detected\n"
        DATA_TYPE=hgemm
        DVAL=4
        if [[ "$(grep -c 'compute_type f16_r' ${LOG})" -gt 0 || "$(grep -c 'compute_type: "f16_r"' ${LOG})" -gt 0 ]]; then
            printf "compute_type of f16 detected, disabling HighPrecisionAccumulate\n"
            DISABLE_HPA=true
        fi
    else
        printf "Could not detect data type in log file, assuming sgemm\n"
        DATA_TYPE=sgemm
        DVAL=2
    fi
fi

if [[ "${GPU}" == mi25 || "${GPU}" == v340 ]]; then
    LIBRARY=vega10
elif [[ "${GPU}" == arcturus ]]; then
    LIBRARY=arcturus
    VERIFICATION=true
elif [[ "${GPU}" == mi50 || "${GPU}" == r7 ]]; then
    GPU=mi50
else
    rocm_agent_enumerator 2>&1 | tee rae.txt
    rocminfo 2>&1 | tee rocminfo.txt
    if [[ "$(grep -c 'gfx900' rae.txt)" -gt 0 ]]; then
        LIBRARY=vega10
        if [[ "$(grep -c 'Compute Unit:            56' rocminfo.txt)" -gt 0 ]]; then
            printf "v340 GPU detected\n"
            GPU=v340
        else
            printf "mi25 GPU detected\n"
            GPU=mi25
        fi
    elif [[ "$(grep -c 'gfx906' rae.txt)" -gt 0 ]]; then
        LIBRARY=vega20
        if [[ "$(grep -c 'Compute Unit:            60' rocminfo.txt)" -gt 0 ]]; then
            printf "mi50 GPU detected\n"
            GPU=mi50
        else
            printf "mi60 GPU detected\n"
            GPU=mi60
        fi
    elif [[ "$(grep -c 'gfx908' rae.txt)" -gt 0 ]]; then
        printf "arcturus GPU detected\n"
        LIBRARY=arcturus
        GPU=arcturus
        VERIFICATION=true

        # currently, only sgemm mfma kernels are supported in automation
        if [[ "${DATA_TYPE}" == sgemm ]]; then
            MFMA=true
        fi
    else
        printf "Could not detect GPU, assuming mi60\n"
        LIBRARY=vega20
        GPU=mi60
    fi
    rm -rf rae.txt rocminfo.txt
fi

if [ -z ${SCLK+foo} ]; then
    rocm-smi 2>&1 | tee rocmsmi.txt
    SCLK=$(cat rocmsmi.txt | awk 'FNR == 6 {print $4}' | cut -d M -f 1)
    printf "SCLK: ${SCLK}\n"
    rm -rf rocmsmi.txt
fi

if [[ "${TENSILE_CLIENT}" != both && "${TENSILE_CLIENT}" != old ]]; then
    printf "Setting Tensile Client to new\n"
    TENSILE_CLIENT=new
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

make_packages()
{
    if [[ $(ls -A *.deb | wc -c) -eq 0 ]]; then
        make package
        cp *.deb ../../../packages/library
    fi
    if [[ $(ls -A clients/*.deb | wc -c) -eq 0 ]]; then
        make package_clients
        cp clients/*.deb ../../../packages/client
    fi
}



SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PROVISION_TUNNING=${SCRIPT_PATH}/provision_tuning.sh

mkdir -p ${OUTPUT_DIR}

if [[ $(ls -A ${OUTPUT_DIR} | wc -c) -eq 0 ]]; then
  if [ -z ${NETWORK+foo} ]; then
    TUNING_ARGS="-z ${LOG}"
  else
    TUNING_ARGS="-d ${LOG} -n ${NETWORK}"
  fi
  TUNING_ARGS+=" -w ${OUTPUT_DIR} -o ${OUTPUT_DIR}.yaml -l ${LIBRARY} --problem-definition ${PROBLEM_DEFINITION} -i ${INITIALIZATION} --client ${TENSILE_CLIENT} "

  if [ ${TILE_AWARE} ]; then
    TUNING_ARGS+=" -a"
  fi

  if [ ${MFMA} ]; then
    TUNING_ARGS+=" -m"
  fi

  if [ ${RK} ]; then
    TUNING_ARGS+=" -r"
  fi

  if [ ${DISABLE_STRIDES} ]; then
    TUNING_ARGS+=" -s"
  fi

  if [ ${DISABLE_HPA} ]; then
    TUNING_ARGS+=" --disable-hpa"
  fi
  ${PROVISION_TUNNING} ${TUNING_ARGS}
fi

# generate the names of the tuning scripts
BUILD_NN=build-${LIBRARY}_${DATA_TYPE}_nn_${OUTPUT_DIR}
BUILD_NT=build-${LIBRARY}_${DATA_TYPE}_nt_${OUTPUT_DIR}
BUILD_TN=build-${LIBRARY}_${DATA_TYPE}_tn_${OUTPUT_DIR}

if [[ "${TUNE_TYPE}" == nn ]]; then
    BUILD_DIRS=${BUILD_NN}
elif [[ "${TUNE_TYPE}" == nt ]]; then
    BUILD_DIRS=${BUILD_NT}
elif [[ "${TUNE_TYPE}" == tn ]]; then
    BUILD_DIRS=${BUILD_TN}
elif [[ "${OMIT_TYPE}" == nn ]]; then
    BUILD_DIRS="${BUILD_NT} ${BUILD_TN}"
elif [[ "${OMIT_TYPE}" == nt ]]; then
    BUILD_DIRS="${BUILD_NN} ${BUILD_TN}"
elif [[ "${OMIT_TYPE}" == tn ]]; then
    BUILD_DIRS="${BUILD_NN} ${BUILD_NT}"
else
    BUILD_DIRS="${BUILD_NN} ${BUILD_NT} ${BUILD_TN}"
fi

TENSILE_PATH=${OUTPUT_DIR}/tensile/Tensile
pushd ${TENSILE_PATH} > /dev/null

TUNING_SH="tune_tensile.sh"
echo "#!/bin/sh" > $TUNING_SH
echo "for dir in $BUILD_DIRS" >> $TUNING_SH
echo "do" >> $TUNING_SH
echo "  cd \${dir}" >> $TUNING_SH
echo "  ./doit.sh > doit-errs 2>&1" >> $TUNING_SH
echo "  cd .." >> $TUNING_SH
echo "done" >> $TUNING_SH

chmod +x $TUNING_SH

# run tuning
./tune_tensile.sh
popd > /dev/null

VERIFICATION_SH="${SCRIPT_PATH}/provision_verification.sh"
VERIFICATION_PATH="valdiate" 
VERIFICATION_SCRIPTS="${OUTPUT_DIR}/scripts"
VERIFICATION_EXE="${VERIFICATION_SH} -w ${VERIFICATION_PATH} -r ${TENSILE_PATH} -l ${LIBRARY}"

if [[ "${LIBRARY}" == arcturus || ("${DATA_TYPE}" == hgemm && "${DISABLE_HPA}" == false) ]]; then
    VERIFICATION_EXE+=" --no-massage"
fi

if [[ "${LIBRARY}" == arcturus ]]; then
    if [[ $(ls -A logs/log-efficiency | wc -c) -eq 0 && "${PUBLIC}" == false ]]; then

      VERIFICATION_EXE+=" --sclk ${SCLK}"
    fi
fi

ROCBLAS_REFERENCE_PATH="${VERIFICATION_PATH}/rocblas/rocBLAS-reference"
ROCBLAS_BENCH_PATH="${ROCBLAS_REFERENCE_PATH}/build/release/clients/staging"

if [ ! -f "${ROCBLAS_BENCH_PATH}/rocblas-bench" ]; then
  ${VERIFICATION_EXE}
fi

RUN_VERIFICATION_SH="${SCRIPT_PATH}/run_validation.sh"
RUN_VERIFICATION_EXE="${RUN_VERIFICATION_SH} -w ${VERIFICATION_PATH} -s ${VERIFICATION_SCRIPTS}"

if [ ! -d "${ROCBLAS_BENCH_PATH}/results_ref" ]; then
  ${RUN_VERIFICATION_EXE}
fi


ANALYSIS_SH="${SCRIPT_PATH}/analyze-results.sh"
ANALYSIS_PATH="${OUTPUT_DIR}/analysis"
ANALYSIS_EXE="${ANALYSIS_SH} -o ${ANALYSIS_PATH} -r ${ROCBLAS_BENCH_PATH}/results_ref -b ${ROCBLAS_BENCH_PATH}/results_validate -z ${VERIFICATION_SCRIPTS}/*-all.sh -f ${SCLK} -s ${DVAL} -g ${GPU} -c ${COUNT} -m ${MFMA}"

if [ ! -d "${ANALYSIS_PATH}" ]; then
    ${ANALYSIS_EXE}
fi
