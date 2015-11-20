


################################################################################
# GenLibrary - Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="CobaltGenBenchmark")
  ap.add_argument("--output-path", dest="outputPath" )
  ap.add_argument("--input-file", dest="inputFiles", action="append" )
  ap.add_argument("--language", dest="language" )
  ap.add_argument("--enable-validation", dest="validate", action="store_true" )

  # parse arguments
  args = ap.parse_args()

  # print settings
  print "CobaltGenBenchmark.py: using language " + args.language

  # generate benchmark
  GenBenchmarkFromFiles( \
      args.inputFiles, \
      args.outputPath, \
      args.language, \
      args.validate )

