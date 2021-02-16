#!/bin/bash

function make_tensile_tuning() {

  local FILE_PATH=$1

  local FILE_NAME; FILE_NAME=$(basename "${FILE_PATH}")
  local FILE_PATH; FILE_PATH=$(ls "$SOURCE/$FILE_NAME")
  local FILE_NAME_NO_EXT="${FILE_NAME%.*}"

  local WORKING_PATH="${DESTINATION}/build-${FILE_NAME_NO_EXT}"

  mkdir -p "${WORKING_PATH}"
  cp "${FILE_PATH}" "${WORKING_PATH}"
  pushd "${WORKING_PATH}" > /dev/null || exit
  {
    echo "#!/bin/sh"
    echo "if [ ! -d 3_LibraryLogic ] || [ -z \"\$(ls -A 3_LibraryLogic)\" ]; then"
    echo "  touch time.begin"
    echo "  ${TENSILE}/Tensile/bin/Tensile ${FILE_NAME} ./ > tuning.out 2>&1"
    echo "  touch time.end"
    echo "fi"
  } > runTensileTuning.sh

  chmod +x runTensileTuning.sh
  popd > /dev/null || exit
}

if [ $# -lt 4 ]; then
  echo "Too few arguments"
  echo "need: SOURCE_PATH DESTINATION_PATH TENSILE_PATH FILE_NAME(s)"
  exit 2
fi

SOURCE=$1
shift
DESTINATION=$1
shift
TENSILE=$1
shift

if [ ! -d "${SOURCE}" ]; then
  echo "The path ${SOURCE} does not exist"
  exit 2
fi

if [ ! -d "${DESTINATION}" ]; then
  echo "The path ${DESTINATION} does not exist"
  exit 2
fi

if [ ! -d "${TENSILE}" ]; then
  echo "The path ${TENSILE} does not exist"
  exit 2
fi

DOIT="${DESTINATION}/runTensileTuning-all.sh"

for CONFIG in "$@"
do
  FILE="${SOURCE}/${CONFIG}"
  if [ ! -f "$FILE" ]; then
    echo "The file ${FILE} does not exist"
    exit 2
  fi
done

DIRS=""
echo "#!/bin/sh" > "${DOIT}"

for CONFIG in "$@"
do
    make_tensile_tuning "${SOURCE}/${CONFIG}"
    DIRNAME="${CONFIG%.*}"
    DIRS+=" ${DIRNAME}"
done

{
  echo "for dir in${DIRS}"
  echo "do"
  echo "  cd build-\${dir} || exit"
  echo "  ./runTensileTuning.sh > tuning-errs.out 2>&1"
  echo "  cd .."
  echo "done"
} >> "${DOIT}"

chmod +x "${DOIT}"
