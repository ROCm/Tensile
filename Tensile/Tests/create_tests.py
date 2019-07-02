#!/usr/bin/python 

# Create a test_py script for all *.yaml files in specified directory
# usage: create_tests.py TEST_DIR
# Run from the Tensile/Tests directory, output script goes in the TEST_DIR/test_TEST_DIR.py

# The directory containing the test script can be passed to pytest:
# PYTHONPATH=. py.test-3 --durations=0 -v Tensile/Tests/TEST_DIR/
from __future__ import print_function
import glob, sys, os

targetDir  = sys.argv[1] if len(sys.argv) > 1 else "."
targetFile = "%s/test_%s.py"%(targetDir,os.path.basename(targetDir))
print("info: writing test script to %s" % targetFile)
outfile = open(targetFile, "w" )
outfile.write("import Tensile.Tensile as Tensile\n\n")
for f in glob.glob("%s/*aml"%targetDir):
    baseName = os.path.basename(f)
    testName = os.path.splitext(baseName)[0]
    if not testName.startswith("test_"):
        testName = "test_" + testName 

    outfile.write ("def %s(tmpdir):\n" % (testName))
    outfile.write (' Tensile.Tensile([Tensile.TensileTestPath("%s"), tmpdir.strpath])\n\n' % (f))

