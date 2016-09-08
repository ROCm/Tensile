SOURCE=main_hsa
KERNEL=sgemm_NT_128x128x8
CLANG=/home/amd/llvm/bin/clang
GCC=/usr/bin/c++

# assemble
echo "assembling kernel"
${CLANG} -x assembler -target amdgcn--amdhsa -mcpu=fiji -c -o kernel.o ${KERNEL}.s

# link
echo "linking kernel"
${CLANG} -target amdgcn--amdhsa kernel.o -o kernel.co

# compile host
echo "compiling host application"
${GCC} -I/opt/rocm/hsa/include -Wall -std=c++11 -g ${SOURCE}.cpp -o ${SOURCE} -rdynamic /opt/rocm/lib/libhsa-runtime64.so -Wl,-rpath,/opt/rocm/lib 

# run application
echo "running host application"
./${SOURCE}
