#!/bin/bash


make_tensile_tuning () {

    local FULLFILENAME=$1

    local BUILDNAME="build"

    local FILENAME=$(basename "$FULLFILENAME")
    local FNAME="${FILENAME%.*}"
    local FILEPATH=$(ls $SOURCE/$FILENAME)

    local WORKINGPATHNAME=${BUILDNAME}-${FNAME}
    local WORKINGPATH=${DESTINATION}/${WORKINGPATHNAME}

    mkdir -p $WORKINGPATH
    cp $FILEPATH $WORKINGPATH
    pushd ${WORKINGPATH} > /dev/null
    echo "#!/bin/sh" > doit.sh
    echo "touch time.begin" >> doit.sh
    echo "../Tensile/bin/Tensile $FILENAME ./ > make.out 2>&1" >> doit.sh
    echo "touch time.end" >> doit.sh

    chmod +x doit.sh
    popd > /dev/null
}

if [ $# -lt 3 ]; then

  echo "Too few arguments"
  echo "need source_path destination_path file_names"
  exit 0

fi

SOURCE="$1"
shift
DESTINATION="$1"
shift

if [ ! -d $SOURCE ]; then
  echo "The path $SOURCE does not exist. Exiting"
  exit 0
fi

if [ ! -d $DESTINATION ]; then
  echo "The path $DESTINATION does not exist. Exiting"
  exit 0
fi

DOIT=$DESTINATION/doit-all.sh

for config in "$@"
do
  FILE="$SOURCE/$config"
  if [ ! -f $FILE ]; then
    echo "The file $FILE does not exist"
    exit 0
  fi
done

DIRS=""
echo "#!/bin/sh" > $DOIT

for config in "$@"
do  
    make_tensile_tuning "${SOURCE}/${config}" 
    DIRNAME="${config%.*}"
    DIRS="${DIRS} ${DIRNAME}"
done

echo "for dir in$DIRS" >> $DOIT
echo "do" >> $DOIT
echo "  cd build-\${dir}" >> $DOIT
echo "  ./doit.sh > doit-errs 2>&1" >> $DOIT
echo "  cd .." >> $DOIT
echo "done" >> $DOIT

chmod +x $DOIT 




