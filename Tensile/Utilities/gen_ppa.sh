#!/bin/bash

###################################################################
#
# Set ATITOOL PATH to atitool directory( can be installed from here
# run tensile_power in STEM directory (Tensile directory)
#  sample command examples
# ./tensile_power -i <yam_file> -d pmlogs -gclk 1300 -mclk 800 --genrpt
# ./tensile_power -i <yam_file> -d pmlogs --genrpt
######################################################################
# expect ATITOOLPATH Environment tool path setup 
# check atitool installed
# input [0] =  yaml file 
# input [1] =  best/all size(s)
# input [2] = output directory
# input [3] = cardId

time_stamp=`date +%y%m%d_%H%M%S`
yaml_filename=""
best_allsizes=1	#1=best perf 0=all 
output_dir=""
card_id=0
gfx_clk=1300
fclk=1225
mclk=800
sclk=972
sudo_cmd="/usr/bin/sudo -S"


#associative array for clk table for vega architecture..
#update requird for different architecture
gfxclk_array=([300]=0 [500]=1 [1000]=2 [1100]=3 [1300]=4 [1500]=5 [1600]=6 [1700]=7 [1800]=8)
socclk_array=([310]=0 [524]=1 [567]=2 [619]=3 [680]=4 [756]=5 [850]=6 [972]=7) 
mclk_array=([167]=0 [350]=1 [600]=2 [800]=3)
fclk_array=([550]=0 [610]=1 [690]=2 [760]=3 [870]=4 [960]=5 [1080]=6 [1225]=7) 

#echo "${gfxclk_array[$gfx_clk]}"

trap "exit" INT TERM ERR
trap "kill 0" EXIT

echo "##########################################################"
echo "The arguments supplied are $*"
echo "##########################################################"


restore_dpm() {
cmd=`${sudo} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmrestore`
}

# error handling
fatal() {
  echo "$0: Error: $1"
  echo ""
  usage
}

error() {
  echo "$0: Error: $1"
  echo ""
  exit 1
}

# usage method
usage() {
  bin_name=`basename $0`
  echo "Tensile power Profiling run script."
  echo "Script requires atitool for collecting power."
  echo "Expects ATITOOL_PATH environment variable set for atitool"
  echo ""
  echo "Usage:"
  echo " tensile_power [-h] [-i <input .yaml file>] [-d  output directory ] [-gclk gpu core clock] [-mclk <memory clock>]"
  echo ""
  echo "Options:"
  echo "  -h - this help"
  echo ""
  echo "  -i <.yaml file> - yaml input file"
  echo "  -d <output directory> - output directory for storing resuls"
  echo "  -all  - measure power for all matrix sizes in yaml"
  echo "  -mclk  - memory clock to program <default = 800>"
  echo "  -gclk  - gpu core clock to program <default = 1300>"
  echo "  -fclk  - fclk clock to program <default = 1225>"
  echo "  -socclk  - soc core clock to program <default = 972>"
  echo "  --genrpt  - Generate report output file pmlog.csv"
  echo ""
  exit 1
}

set_dpm_and_run() {
 dpm="$1"
 gfx_level="$2"
 mclk_level="$3"
 fclk_level="$4"
 soc_level="$5"
 log_tag=''
  if [ "$dpm" = "force" ] ; then
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=$card_id` 
    if [ -z "${cmd}" ]; then
        echo "${ATITOOL_PATH}/atitool -i=$card_id Failed"
    fi
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=gfx,${gfx_level}`
    if [ -z "${cmd}" ]; then
        echo "${ATITOOL_PATH}/atitool -i=$card_id -ppdpmforce=gfx,${gfx_level} Failed"
    fi
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=mclk,${mclk_level}`
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=pcie,1`
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=fclk,${fclk_level}`
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=soc, ${soc_level}`
  elif [ "$dpm" -eq "on" ]; then
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id}  -ppdpmrestore`
  fi 
 cmd=`sleep 5s` # let clk force take effect
 log_tag="dpmForce_gfx${gfx_clk}_mclk${mclk}_socclk${soc_clk}_${time_stamp}"

 echo "#####################################################################"
 echo "Start background process atitool for capturing power data....."
 echo "#####################################################################"
 cmd="${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -pmlogall -pmperiod=50 -pmstopcheck -pmoutput=${output_dir}/${log_tag}_pmlog.csv"
 eval "${cmd}" &>/dev/null &disown
 tensile_cmd="HIP_VISIBLE_DEVICES=${card_id} HSA_ENABLE_SDMA=1 ./Tensile/bin/Tensile ${yaml_filename} ${output_dir} 2>&1 | tee ${output_dir}/${log_tag}.log"
 eval "${tensile_cmd}" 
}

generate_report=0
ARG_IN=""
while [ 1 ] ; do
  ARG_IN=$1
  ARG_VAL=1
  if [ "$1" = "-h" ] ; then
    usage
  elif [ "$1" = "-i" ] ; then
    yaml_filename="$2"
  elif [ "$1" = "-d" ] ; then
    output_dir="$2"
  elif [ "$1" = "-gclk" ] ; then
    gfx_clk="$2"
  elif [ "$1" = "-mclk" ] ; then
    mclk="$2"
  elif [ "$1" = "-socclk" ] ; then
    soc_clk="$2"
  elif [ "$1" = "-fclk" ] ; then
    fclk="$2"
  elif [ "$1" = "--all" ] ; then
    best_allsizes=0
    ARG_VAL=0
  elif [ "$1" = "--genrpt" ] ; then
    generate_report=1
    ARG_VAL=0
  else
    break
  fi
  shift
  if [ "$ARG_VAL" = 1 ] ; then shift; fi
done

ARG_CK=`echo $ARG_IN | sed "s/^-.*$/-/"`
if [ "$ARG_CK" = "-" ] ; then
  fatal "Wrong option '$ARG_IN'"
fi

if [ -z "$yaml_filename" ] ; then
  fatal "Need input file"
fi

if [ -z "$output_dir" ] ; then
   output_dir="./tensile_pmlogs"
fi

## check ATITOOLPATH variable
if [ -z "${ATITOOL_PATH}" ]; then
    fatal "ATITOOL_PATH is not set"
else
    echo "ATITOOL_PATH is set: $ATITOOL_PATH"
fi

echo "##########################################################"
echo "Creating output directory $output_dir......"
echo "##########################################################"
mkdir -p "${output_dir}"

#for i in 4 do
set_dpm_and_run "force" ${gfxclk_array[$gfx_clk]} ${mclk_array[$mclk]} ${fclk_array[$fclk]} ${socclk_array[$sclk]}
#end

##genrate report calling
result_file="${output_dir}/pmlog.csv"
FILES=./${output_dir}/*.log
newline=$'\n' 

if [ "${generate_report}" = 1 ]; then
echo "##########################################################"
echo "Generating consolidated output pmlog.csv ....."
echo "##########################################################"

cols="KernelName,ProblemSize,Fset (MHz),Peak Power (W),Power VDDCR GFX,Power VDDCR SOC,Power VDDIO MEM,Power VDDCI MEM,Temperature C,Fload (MHz),Measured GFlops,Software Efficiency (%),Silicon Efficiency (%),Overall Efficiency (%)"
echo  "${cols}" > ${result_file}

for f in $FILES 
do
  echo "--------------------------------------------"
  echo "$(basename "$f")"

  peak_tflops=0
  peak_gflops=`cat "$f"  | grep Fastest | awk '{print $2}'`
  fastkernel_name=`cat $f  | grep Fastest | awk '{print $8}'`
  problem_size=`cat $f  | grep -Ei 'Problem.*: [0-9]+,' | awk '{print $2,$3,$5}'`
  #cmd = "cat #{f}  | grep PASSED | awk '{print $3 }'"
  #kernel_name = `#{cmd}`
  kernel_name=${fastkernel_name%$newline}
  peak=4096 
  if [[ "$kernel_name" =~ "_DB_" ]]
  then
     peak=4096
  fi
  reg_exp="_SB_" 
  if [[ "$kernel_name" =~ "_SB_" ]] 
  then
     peak=8192
  fi
  if [[ "$kernel_name" =~ "_HB_" ]]
  then
     peak=`echo "8192*2" | bc`
  fi
  filename=$(basename -- "$f")
  filename=`echo ${filename%%.*}`
  pm_file="./${output_dir}/${filename}_pmlog.csv"

  echo "$pm_file"
# power vddcr_gfx: Z (26)
# power vddcr_soc: AH (26 + 8 = 34)
# power vddio_mem: AO (26 + 15 = 41)
# power vddci_mem: AV (26 + 22 = 48)

  #tmp = `cat #{pm_file} | cut -d ',' -f 3,6,7,20,109,110`.split("\n")
  #tmp_array=($(cat ${pm_file} | cut -d ',' -f 3,6,7,20,26,34,41,48,109,110))
  cmd="cat ${pm_file} | cut -d ',' -f 3,6,7,20,26,34,41,48,109,110"
  eval "${cmd}" > "./${output_dir}/tmpFile"
  #targets=($(grep -HRl "pattern" .))
  #sorted_array=($(cat ./${output_dir}/tmpFile | sort -t, -k1,1 -nr))
  cmd="sort -t, -k1,1 -nr ${output_dir}/tmpFile"
  eval "${cmd}" > "./${output_dir}/outFile"
  cmd=`rm -f ./${output_dir}/tmpFile`
  read -r peak_line < ./${output_dir}/outFile
  echo "peak power measured: ${peak_line}"
  cmd=`rm -f ./${output_dir}/outFile`
  IFS=','
  read -ra elements <<< "$peak_line"
  peak_power=${elements[0]}
  temperature=${elements[1]}
  hbm_tmp=${elements[2]}
  vddgfx=${elements[3]}
  power_vddcr_gfx=${elements[4]}
  power_vddcr_soc=${elements[5]}
  power_vddio_mem=${elements[6]}
  power_vddci_mem=${elements[7]}
  fset=${elements[8]}
  fload=${elements[9]}
  unset IFS

  measured_gflops=`echo "scale=2; $peak_gflops/1" | bc`
  software_eff=`echo "scale=2; (100*${peak_gflops}/(($peak*$fload)/1000))" | bc`
  silicon_eff=`echo "scale=2; (100*$fload/$fset)" | bc`
  overall_eff=`echo "(100*$peak_gflops/($peak*$fset/1000))" | bc`

  gflops_fload=`echo "($peak*$fload)/1000" | bc`
  gflops_fset=`echo "($peak*$fset)/1000" | bc`

_size=${problem_size%$newline}
delimeter=$','
_size=${problem_size%$delimeter}

  echo "kernel_name: ${kernel_name}" 
  echo "problem_size: ${_size}" 
  echo "peak power: ${peak_power}"
  echo "vddgfx: ${vddgfx}"
  echo "temperature: ${temperature}"
  echo "hbm temperature: ${hbm_tmp}"
  echo "fload: ${fload}"
  echo "gflops fload: ${gflops_fload}"
  echo "gflops fset:  ${gflops_fset}"
  echo "silicon eff: ${silicon_eff}"
  echo "overall eff: ${overall_eff}"
  echo "software eff: ${software_eff}"

  echo "${kernel_name},${_size},${fset},${peak_power},${power_vddcr_gfx},${power_vddcr_soc},${power_vddio_mem},${power_vddci_mem},${fload},${gflops_fload},${measured_gflops},${software_eff},${silicon_eff},${overall_eff}" >>  ${result_file}
done

fi

exit 1
