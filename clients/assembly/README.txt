at the top of run.sh choose between 3 assembly files
you'll probably needs to change the paths to the compilers, too

fmac_only
 - loop with 512 fmac's and as little else as possible
 - 98.7% efficient

read_global
 - loop with 512 fmac's and reads from global memory associated with gemm
 - 96.6% efficient

read_lds
 - loop with 512 fmac's and reads from local memory associated with gemm
 - 97.1% efficient 11 reads/iter
 - 94.1% efficient >=12 reads/iter
