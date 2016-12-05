import Common
import Structs

from Common import printDebug

def benchmarkProblemType( config ):
  printDebug(1,"Tensile::BenchmarkProblemType %s" % str(config) )

def main(  config ):
  printDebug(1,"BenchmarkProblems::main")
  for problemType in config:
    benchmarkProblemType(problemType)

  pass
