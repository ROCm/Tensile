#!/bin/bash

# This script needs to be edited to bump version for new release.
# Version will be bumped in Tensile/__init__.py and in .yaml files

OLD_VERSION="4.10.0"
NEW_VERSION="4.11.0"

OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.2"
NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.8.0"

sed -i "s/${OLD_VERSION}/${NEW_VERSION}/g" Tensile/__init__.py
sed -i "s/${OLD_VERSION}/${NEW_VERSION}/g" HostLibraryTests/CMakeLists.txt

echo "The version number also needs to be fixed in Tensile/cmake/TensileConfigVersion.cmake ."

#only update when there is a major version change
#for FILE in Tensile/Configs/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#done
