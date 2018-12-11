A set of scripts used for staging and runing the tensile tuning.



PeformanceAnalysis.py takes a path containing a set of performance results for rocblas-benchmark
and produces an aggregated results csv

the inputs files are numbered testname.*

example:

hpa_hgemm_resnet50_list1.1
hpa_hgemm_resnet50_list1.2
hpa_hgemm_resnet50_list1.3
hpa_hgemm_resnet50_list1.4

usage: PeformanceAnalysis.py [-h] input_path output_path

positional arguments:
  input_path   path where the results are located
  output_path  path where the processed files are to go

optional arguments:
  -h, --help   show this help message and exit



CompareResults.py takes two aggregated performance results (current and new) and produces a csv with the comparesons

usage: CompareResults.py [-h] current_file new_file combined_file

positional arguments:
  current_file   path where the current results are located
  new_file       path where the new files are located
  combined_file  path where the combined results are located

optional arguments:
  -h, --help     show this help message and exit


TensileExtractSizes.py inputs a Tensile LogicFile and write out all the sizes

usage: TensileExtractSizes.py [-h] ExactLogicPath OutputPath

positional arguments:
  ExactLogicPath  Path to the exact LibraryLogic.yaml input files.
  OutputPath      Where to write library files?

optional arguments:
  -h, --help      show this help message and exit



script/stage_tuning.sh 

takes a source directory a destination directory and a list of Tensile configuration files and populates the build area to set up
Tensile tuning. 

the command
stage_tuning.sh Tensile/Configs Tensile rocblas_hgemm_asm_lite.yaml rocblas_sgemm_asm_lite.yaml

will create a doit-all.sh script in the Tensile directory all with two directories build-rocblas_hgemm_asm_lite and build-rocblas_sgemm_asm_lite

the ./doit-all.sh will launch Tensile for each of the configuration yamls. each sub directory contains a doit.sh, which excecutes tensile for the config, and the configuration file corresponding to the directory name. 

















