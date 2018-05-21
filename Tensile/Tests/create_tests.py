# Create a test_py script for all test*yaml files in specified directory
# usage: create_tests.py DIR_NAME
# Create a test script that can be passed to pytest:
# PYTHONPATH=. py.test -v Tensile//Tests/MY_TEST_DIR/

import glob, sys, os

targetDir  = sys.argv[1] if sys.argv > 1 else "."
targetFile = "%s/test_%s.py"%(targetDir,os.path.basename(targetDir))
print "info: writing test script to %s" % targetFile
outfile = open(targetFile, "w" )
outfile.write("import Tensile.Tensile as Tensile\n\n")
for f in glob.glob("%s/test*yaml"%targetDir):
    baseName = os.path.basename(f)
    testName = os.path.splitext(baseName)[0]
    outfile.write ("def %s(tmpdir):\n" % (testName))
    outfile.write (' Tensile.Tensile([Tensile.TensileTestPath("%s"), tmpdir.strpath])\n\n' % (f))

