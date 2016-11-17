#rocm-smi --setsclk 7
#sleep 1
#rocm-smi -a

SOURCE=main
#KERNEL=fmac_only
KERNEL=read_global
#KERNEL=read_lds
#KERNEL=sgemm_NT_working
CLANG=/home/amd/llvm/bin/clang
GCC=/usr/bin/c++

# assemble
echo "assembling kernel"
${CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -c -o ${KERNEL}.o ${KERNEL}.s

# link
echo "linking kernel"
${CLANG} -target amdgcn--amdhsa ${KERNEL}.o -o ${KERNEL}.co

# compile host
echo "compiling host application"
${GCC} -I/opt/rocm/hsa/include -Wall -std=c++11 -DKERNEL_FILE_NAME=\"${KERNEL}.co\" ${SOURCE}.cpp -o ${SOURCE} -rdynamic /opt/rocm/lib/libhsa-runtime64.so -Wl,-rpath,/opt/rocm/lib 

# run application
echo "running host application"
./${SOURCE}

#rocm-smi --resetclocks
