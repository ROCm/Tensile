#!/bin/sh


SCRIPT=./make_tensile_tuning.sh

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
    $SCRIPT $SOURCE $DESTINATION "$SOURCE/$config" 
    DIRNAME="${config%.*}"
    DIRS="$DIRS $DIRNAME"
done

echo "for dir in$DIRS" >> $DOIT
echo "  cd build-\${dir}" >> $DOIT
echo "  ./doit.sh > doit-errs 2>&1" >> $DOIT
echo "  cd .." >> $DOIT
echo "done" >> $DOIT

chmod +x $DOIT 





