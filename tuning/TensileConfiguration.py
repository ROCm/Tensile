

import os
import sys
import argparse


################################################################################
# Print Debug
################################################################################

#def printWarning(message):
#  print "Tensile::WARNING: %s" % message
#  sys.stdout.flush()

def printExit(message):
  print "Tensile::FATAL: %s" % message
  sys.stdout.flush()
  sys.exit(-1)

try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

#HR = "################################################################################"


def ensurePath( path ):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

################################################################################
# Define Constants
################################################################################

def constant(f):
  def fset(self, value):
    raise TypeError
  def fget(self):
    return f(self)
  return property(fget, fset)

class _Const(object):
  @constant
  def GlobalParameters(self):
    return "GlobalParameters"
  
  @constant
  def BenchmarkProblems(self):
    return "BenchmarkProblems"

  @constant
  def LibraryLogic(self):
    return "LibraryLogic"

  @constant
  def LibraryClient(self):
    return "LibraryClient"

CONST = _Const()


################################################################################
# Tuning Configuration Container
################################################################################
class TuningConfiguration:
  
  def __init__(self,filename=None):
    print "implement"
    if filename is not None:
      print ("# Reading configuration: " + filename)
      try:
        stream = open(filename, "r")
      except IOError:
        printExit("Cannot open file: %s" % filename )
      
      data = yaml.load(stream, yaml.SafeLoader)

      if CONST.GlobalParameters in data:
        self.__set_globalParameters(data[CONST.GlobalParameters])
      else:
        self.__set_globalParameters(None)
        
      if CONST.BenchmarkProblems in data:
        self.__set_benchmarkProblems(data[CONST.BenchmarkProblems])
      else:
        self.__set_benchmarkProblems(None)
        
      if CONST.LibraryLogic in data:
        self.__set_libraryLogic(data[CONST.LibraryLogic])
      else:
        self.__set_libraryLogic(None)

      if CONST.LibraryClient in data:
        self.__set_libraryClient(data[CONST.LibraryClient])
      else:
        self.__set_libraryClient(None)

      stream.close()

    else:
      self.__set_globalParameters(None)
      self.__set_benchmarkProblems(None)
      self.__set_libraryLogic(None)
      self.__set_libraryLogic(None)


  def __get_globalParameters(self):
    return self.__globalParameters

  def __set_globalParameters(self,value):
    self.__globalParameters = value
  
  globalParamters = property(__get_globalParameters,__set_globalParameters)

  def __get_benchmarkProblems(self):
    return self.__benchmarkProblems

  def __set_benchmarkProblems(self,value):
    self.__benchmarkProblems = value
  
  benchmarkProblems = property(__get_benchmarkProblems,__set_benchmarkProblems)

  def __get_libraryLogic(self):
    return self.__libraryLogic

  def __set_libraryLogic(self,value):
    self.__libraryLogic = value

  libraryLogic = property(__get_libraryLogic,__set_libraryLogic)

  def __get_libraryClient(self):
    return self.__libraryClient 

  def __set_libraryClient(self,value):
    self.__libraryClient = value

  libraryClient = property(__get_libraryClient,__set_libraryClient)
  
  
  
