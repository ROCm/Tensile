#!/bin/bash

# the convention is master has even leading digit, and develop has odd leading digit
#
# Note that for changes in minor version number it may not be necessary to update
# MinimumRequiredVersion in .yaml files, it may only be necessary to update __init__.py
#
# It is necessary to get the even number for master branch and Odd number for 
# develop branch correct. This applies also to .yaml files

OLD_VERSION="5.0.1"
NEW_VERSION="4.0.2"

OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 5.0.1"
NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.0.2"

sed -i "s/${OLD_VERSION}/${NEW_VERSION}/g" Tensile/__init__.py

for FILE in Tensile/Configs/*yaml
do
  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
done
