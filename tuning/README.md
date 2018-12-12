

## Tensile tuning process for rocBLAS

#### Tensile tuning step

See the [Tensile tuning](https://github.com/ROCmSoftwarePlatform/Tensile/wiki) wiki for how the tuning works. The staging area we use for the our process is withing build directors in the root level of the Tensile clone. This step will produce LibraryLogic files within each for the build subdirectories. `../build-hgemm_asm_full/3_LibraryLogic`.

#### Building rocBLAS with new configurations

The updated configurations get placed in the rocBLAS path ` rocBLAS/library/src/blas3/Tensile/Logic/asm_full`. They can either go in raw if a complete tuning was targeted or the configuration can be merged using the Tensile merging tool, `Tensile/Utilities/merge_rocblas_yaml_files.py`.

#### Build rocBLAS

run `./install -c` in the rocBLAS source directory. See [rocBLAS build](https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/1.Build) for more information.

#### Performance measurements

The performance can be measure using rocblas-bench.

Run benchmarks with the following:

`cd [BUILD_DIR]/release/clients/staging`
`./rocblas-bench -h`

The following are examples for running particular gemm and gemv benchmark:

`./rocblas-bench -f gemm -r s -m 1024 -n 1024 -k 1024 --transposeB T -v 1`
`./rocblas-bench -f gemv -m 9216 -n 9216 --lda 9216 --transposeA T`



## Tensile Automation Scripts

#### Analyze benchmark results

The PeformanceAnalysis.py takes a path containing a set of performance results for rocblas-benchmark and produces an aggregated results in a csv file. For this to work the inputs files are numbered testname.*

example.

`hpa_hgemm_resnet50_list1.1`  
`hpa_hgemm_resnet50_list1.2`  
`hpa_hgemm_resnet50_list1.3`  
`hpa_hgemm_resnet50_list1.4`


```
usage: 

PeformanceAnalysis.py [-h] input_path output_path

positional arguments:
  input_path   path where the results are located
  output_path  path where the processed files are to go

optional arguments:
  -h, --help   show this help message and exit
```


CompareResults.py takes two aggregated performance results (current and new) and produces a csv with the comparisons in the same csv file.

```
usage: 

CompareResults.py [-h] current_file new_file combined_file

positional arguments:
  current_file   path where the current results are located
  new_file       path where the new files are located
  combined_file  path where the combined results are located

optional arguments:
  -h, --help     show this help message and exit
```

#### Operational Scripts

TensileExtractSizes.py inputs a Tensile LogicFile and write out all the sizes

```
usage: 

TensileExtractSizes.py [-h] ExactLogicPath OutputPath

positional arguments:
  ExactLogicPath  Path to the exact LibraryLogic.yaml input files.
  OutputPath      Where to write library files?

optional arguments:
  -h, --help      show this help message and exit
```

##### for staging the tuning process

The `script/stage_tuning.sh` takes a source directory a destination directory and a list of Tensile configuration files and populates the build area to set up
Tensile tuning. 

```
Usage:

stage_tuning.sh InputConfigPath BuildPath file1.yaml file2.yaml ...


InputConfigPath: is the path where the tensile configurations are located
BuildPath: this the staging area for the tuning, usually the Tensile root path

file?.yaml: the configuration files that are being populated

```

example:

`stage_tuning.sh Tensile/Configs Tensile rocblas_hgemm_asm_lite.yaml rocblas_sgemm_asm_lite.yaml`

This will create a doit-all.sh script in the Tensile directory all with two directories build-rocblas_hgemm_asm_lite and build-rocblas_sgemm_asm_lite

the ./doit-all.sh will launch Tensile for each of the configuration yamls. each sub directory contains a doit.sh, which executes tensile for the config, and the configuration file corresponding to the directory name. 

















