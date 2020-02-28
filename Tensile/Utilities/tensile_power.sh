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
gfx_clk=1100
mclk=800
soc_clk=1000
sudo_cmd="/usr/bin/sudo -S"

#associative array for clk table for vega architecture..
#update requird for different architecture
gfxclk_array=([300]=0 [500]=1 [1000]=2 [1100]=3 [1300]=4 [1500]=5 [1600]=6 [1700]=7 [1800]=8)
socclk_array=([300]=0 [400]=1 [500]=2 [600]=3 [700]=4 [800]=5 [900]=6 [1000]=7) 
mclk_array=([167]=0 [350]=1 [600]=2 [800]=3)

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
  echo "  -socclk  - soc core clock to program <default = 1000>"
  echo "  --genrpt  - Generate report output file pmlog.csv"
  echo ""
  exit 1
}

set_dpm_and_run() {
 dpm="$1"
 gfx_level="$2"
 mclk_level="$3"
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
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=fclk,7`
    cmd=`${sudo_cmd} ${ATITOOL_PATH}/atitool -i=${card_id} -ppdpmforce=soc,7`
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
 tensile_cmd="HIP_VISIBLE_DEVICES=${card_id} HSA_ENABLE_SDMA=1  python ./Tensile/Tensile.py ${yaml_filename} build_pmrun 2>&1 | tee ${output_dir}/${log_tag}.log"
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
set_dpm_and_run "force" ${gfxclk_array[$gfx_clk]} ${mclk_array[$mclk]}
#end

##genrate report calling

echo "##########################################################"
echo "Generating consolidated output pmlog.csv ....."
echo "##########################################################"
if [ "${generate_report}" = 1 ]; then
   cmd=`ruby ./Tensile/Utilities/gen_csv.rb  ${output_dir} ${best_allsizes}`
fi

exit 1
