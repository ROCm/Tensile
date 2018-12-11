#!/bin/sh

if [ $# -lt 3 ]; then

  echo "Too few arguments"
  echo "need source_path destination_path file_name [build_name]"
  exit 0

fi

SOURCE=$1
DESTINATION=$2
FULLFILENAME=$3

if [ ! -d $SOURCE ]; then
  echo "The path $SOURCE does not exist. Exiting"
  exit 0
fi

if [ ! -d $DESTINATION ]; then
  echo "The path $DESTINATION does not exist. Exiting"
  exit 0
fi

CURRENTPATH=$(pwd)
BUILDNAME="build"
FILENAME=$(basename "$FULLFILENAME")
FNAME="${FILENAME%.*}"
#EXT="${FILENAME##*.}"
FILEPATH=$(ls $SOURCE/$FILENAME)

WORKINGPATHNAME=${BUILDNAME}-${FNAME}
WORKINGPATH=${DESTINATION}/${WORKINGPATHNAME} 


mkdir -p $WORKINGPATH
cp $FILEPATH $WORKINGPATH
cd $WORKINGPATH

echo "#!/bin/sh" > doit.sh
echo "rocm-smi -d1 --setsclk 4 --setmclk 3" >> doit.sh
echo "sudo ~zaliu/bin/ipmicfg -fan 1" >> doit.sh
echo "touch time.begin" >> doit.sh
echo "python ../Tensile/Tensile.py $FILENAME ./ > make.out 2>&1" >> doit.sh
echo "touch time.end" >> doit.sh
echo "sudo ~zaliu/bin/ipmicfg -fan 2" >> doit.sh

chmod +x doit.sh

cd $CURRENTPATH


