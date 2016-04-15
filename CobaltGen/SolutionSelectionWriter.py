import Structs
import SolutionWriter
import copy

class SolutionSelectionWriter:

  def __init__(self, psMap, backend):
    self.backend = backend
    self.solutionWriter = SolutionWriter.SolutionWriter(backend)
    self.psMap = psMap
    self.tolerance = 0.05
    self.kernelSet = set()
    self.solutionSet = set()
  
  #############################################################################
  # write top level getSolution
  # chooses amongst devices
  #############################################################################
  def writeGetSolutionTop(self):
    functionName = "getSolutionTop"
    # source file
    s = ""
    s += "#include \"Problem.h\"\n"
    s += "#include \"CobaltGetSolution.h\"\n"
    for deviceProfile, exactMatches in self.psMap.iteritems():
      s += "#include \"CobaltGetSolution_" + deviceProfile.libString() + ".h\"\n"
    s += "\n"
    s += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status ) {\n"
    # if match device
    for deviceProfile, exactMatches in self.psMap.iteritems():
      s += "  if ( problem->pimpl->deviceProfile.numDevices() == " + str(len(deviceProfile.devices)) + " ) {\n"
      s += "    if ( problem->pimpl->deviceProfile[0].matches(\"" + deviceProfile.devices[0].name + "\")"
      for i in range(1, len(deviceProfile.devices)):
        s += " && problem->pimpl->deviceProfile[" + str(i) + "].matches(\"" + deviceProfile.devices[i] + "\")"
      s += "   ) {\n"
      s += "      return getSolution_" + deviceProfile.libString() + "(problem, status);\n"
      s += "    }\n"
      s += "  }\n"
    # else doesn't match any device
    for deviceProfile, exactMatches in self.psMap.iteritems():
      s += "  /* doesn't match any known device; return a default */\n"
      s += "  return getSolution_" + deviceProfile.libString() + "(problem, status);\n"
    s += "}\n"
    s += "\n"

    # header file
    h = ""
    h += "#ifndef COBALT_GETSOLUTION_H\n"
    h += "#define COBALT_GETSOLUTION_H\n"
    h += "\n"
    h += "#include \"Cobalt.h\"\n"
    h += "\n"
    h += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"

    return (s, h)

  
  #############################################################################
  # write device-level getSolution
  # chooses amongst exact matches
  # TODO - in the future, we need another level of selection right after this one
  # which matched the free index order (sorted by strideA+strideB) and summation
  # index order (sorted by same) because that is an optimization which matters
  # to higher dimensions
  #############################################################################
  def writeGetSolutionForDevice( self, deviceProfile, exactMatches):
    functionName = "getSolution_" + deviceProfile.libString()
    s = ""
    s += "#include \"Problem.h\"\n"
    s += "#include \"CobaltGetSolution_" + deviceProfile.libString() + ".h\"\n"
    for exactMatch, problems in exactMatches.iteritems():
      s += "#include \"CobaltGetSolution_" + exactMatch.libString() + ".h\"\n"
    s += "\n"
    s += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status ) {\n"
    s += "  bool problemRequiresLeadingStrides;\n"
    s += "\n"
    
    for exactMatch, problems in exactMatches.iteritems():
      # if problem exactly matches EXACT_MATCH
      s += "  problemRequiresLeadingStrides = problem->pimpl->tensorC[0].stride != 1 || problem->pimpl->tensorA[0].stride != 1 || problem->pimpl->tensorB[0].stride != 1;\n"
      s += "  if ( problem->pimpl->getDataTypeC() == " + exactMatch.typeC.getLibString() + "\n"
      s += "      && problem->pimpl->getDataTypeA() == " + exactMatch.typeA.getLibString() + "\n"
      s += "      && problem->pimpl->getDataTypeB() == " + exactMatch.typeB.getLibString() + "\n"
      s += "      && problem->pimpl->getDataTypeAlpha() == " + exactMatch.typeAlpha.getLibString() + "\n"
      s += "      && problem->pimpl->getDataTypeBeta() == " + exactMatch.typeBeta.getLibString() + "\n"
      s += "      && problem->pimpl->operationType == " + exactMatch.operationType.getLibString() + "\n"
      s += "      && (!problem->pimpl->useOffsets || problem->pimpl->useOffsets == " + ("false" if exactMatch.ppdOffsets else "true") + ")\n"
      s += "      && (!problemRequiresLeadingStrides || problemRequiresLeadingStrides == " + ("false" if exactMatch.ppdLeadingStride else "true") + ")\n"
      s += "      && problem->pimpl->indicesFree.size() == " + str(exactMatch.numIndicesFree) + "\n"
      s += "      && problem->pimpl->indicesA.size() == " + str(len(exactMatch.indexAssignmentsA)) + "\n"
      s += "      && problem->pimpl->indicesB.size() == " + str(len(exactMatch.indexAssignmentsB)) + " ) {\n"
      
      s += "    if ("
      s += " problem->pimpl->indicesA[0] == " + str(exactMatch.indexAssignmentsA[0]) + "\n"
      for i in range(1, len(exactMatch.indexAssignmentsA)):
        s += "        && problem->pimpl->indicesA[" + str(i) + "] == " + str(exactMatch.indexAssignmentsA[i]) + "\n"
      s += "        && problem->pimpl->indicesB[0] == " + str(exactMatch.indexAssignmentsB[0]) + "\n"
      for i in range(1, len(exactMatch.indexAssignmentsB)):
        s += "        && problem->pimpl->indicesB[" + str(i) + "] == " + str(exactMatch.indexAssignmentsB[i])
        if i == len(exactMatch.indexAssignmentsB)-1:
          s += " ) {\n"
        else:
          s += "\n"

      s += "      return getSolution_" + exactMatch.libString() + "( problem, status);\n"
      s += "    }\n"
      s += "  }\n"
    
    s += "  *status = cobaltStatusProblemNotSupported;\n"
    s += "  return nullptr;\n"
    s += "}\n"
    s += "\n"
    
    # header file
    h = ""
    h += "#ifndef COBALT_" + functionName.upper() + "_H\n"
    h += "#define COBALT_" + functionName.upper() + "_H\n"
    h += "\n"
    h += "#include \"Cobalt.h\"\n"
    h += "\n"
    h += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"
    return (s, h)
  
  # fallback problem/solution pair = "b" solution or "m" solution which launched multiple kernels
  def isFallback(self, problem, solution):
    if solution.branch[0].isBranched() and solution.branch[1].isBranched():
      return True
    if solution.branch[0].isMultiple():
      problemSize0 = problem.tensorC.dimensions[solution.kernels[0].indexAssignmentDim0].size
      tileSize0 = solution.kernels[0].tile.workGroup[0] * solution.kernels[0].tile.microTile[0]
      if problemSize0 % tileSize0 > 0:
        return True
    if solution.branch[1].isMultiple():
      problemSize1 = problem.tensorC.dimensions[solution.kernels[0].indexAssignmentDim1].size
      tileSize1 = solution.kernels[0].tile.workGroup[1] * solution.kernels[0].tile.microTile[1]
      if problemSize1 % tileSize1 > 0:
        return True
    return False
  
  # single-kernel problem/solution pair = "b" solution or "m" solution which launched only one kernel
  def isSingleton(self, problem, solution):
    if solution.branch[0].isMultiple():
      problemSize0 = problem.tensorC.dimensions[solution.kernels[0].indexAssignmentDim0].size
      tileSize0 = solution.kernels[0].tile.workGroup[0] * solution.kernels[0].tile.microTile[0]
      if problemSize0 % tileSize0 > 0:
        return False
    if solution.branch[1].isMultiple():
      problemSize1 = problem.tensorC.dimensions[solution.kernels[0].indexAssignmentDim1].size
      tileSize1 = solution.kernels[0].tile.workGroup[1] * solution.kernels[0].tile.microTile[1]
      if problemSize1 % tileSize1 > 0:
        return False
    return True

  
  def getSize(self, problem):
    totalSize = 1
    # multiply free indices
    for dimension in problem.tensorC.dimensions:
      totalSize *= dimension.size
    # multiply summation indices
    for i in range(0, len(problem.operation.indexAssignmentsA)):
      index = problem.operation.indexAssignmentsA[i]
      inC = index < len(problem.tensorC.dimensions)
      inB = index in problem.operation.indexAssignmentsB
      if inB and not inC: # is summation dimension
        totalSize *= problem.tensorA.dimensions[i].size
    return totalSize

  def getGFlops(self, problem, timeMS):
    totalFlops = self.getSize(problem)
    if problem.tensorA.dataType.isReal():
      totalFlops *= 2
    else:
      totalFlops *= 8
    gFlops = (totalFlops/1000000000.0) / (timeMS/1000.0)
    return gFlops

  def getFallbacks(self, problemSolutionPairs):
    fallbacks = []
    for i in range(0, len(problemSolutionPairs)):
      problem = problemSolutionPairs[i][0]
      solution = problemSolutionPairs[i][1]
      if self.isFallback(problem, solution):
        fallbacks.append( problemSolutionPairs[i] )
    return fallbacks
  
  def getSingletons(self, problemSolutionPairs):
    singles = []
    for i in range(0, len(problemSolutionPairs)):
      problem = problemSolutionPairs[i][0]
      solution = problemSolutionPairs[i][1]
      if self.isSingleton(problem, solution):
        singles.append( problemSolutionPairs[i] )
    return fallbacks

  def getIndexOfFastest( self, psps ):
    fastestIndex = 0
    fastestProblem = psps[fastestIndex][0]
    fastestSolution = psps[fastestIndex][1]
    fastestTime = psps[fastestIndex][2]
    fastestGFlops = self.getGFlops(fastestProblem, fastestTime)
    for i in range(1, len(psps)):
      if self.getGFlops(psps[i][0], psps[i][2]) > fastestGFlops:
        fastestIndex = i
        fastestProblem = psps[fastestIndex][0]
        fastestSolution = psps[fastestIndex][1]
        fastestTime = psps[fastestIndex][2]
        fastestGFlops = self.getGFlops(fastestProblem, fastestTime)
    return fastestIndex

  def getIndexOfLargest( self, psps ):
    largestIndex = 0
    largestProblem = psps[largestIndex][0]
    largestSize = self.getSize(largestProblem)
    for i in range(1, len(psps)):
      if self.getSize(psps[i][0]) > largestSize:
        largestIndex = i
        largestProblem = psps[largestIndex][0]
        largestSize = self.getSize(largestProblem)
    return largestIndex

  def sortSizePSPs( self, inputPSPs ):
    psps = copy.deepcopy(inputPSPs)
    s = []
    while len(psps) > 0:
      indexOfLargest = self.getIndexOfLargest(psps)
      s.append( psps.pop(indexOfLargest) )
    return s

  def sortSpeedPSPs( self, inputPSPs ):
    psps = copy.deepcopy(inputPSPs)
    s = []
    while len(psps) > 0:
      indexOfFastest = self.getIndexOfFastest(psps)
      s.append( psps.pop(indexOfFastest) )
    return s

  def getPSPsWithSize(self, psps, size ):
    s = []
    for psp in psps:
      self.sizeOfPSP = self.getSize(psp[0])
      if self.sizeOfPSP == size:
        s.append(psp)
    return s

  def getIndexOfSolution( self, psps, solution ):
    for i in range(0, len(psps)):
      if psps[i][1] == solution:
        return i
    return len(psps)

  # returns a problem with largest size
  def getLargestSize(self, psps):
    largestSize = 0;
    largestProblem = psps[largestSize][0]
    for psp in psps:
      size = self.getSize(psp[0])
      if size > largestSize:
        largestSize = size
        largestProblem = psp[0]
    return largestProblem

  #  1 if p0 > p1
  # -1 is p0 < p1
  #  0 if equal
  def compareSize(self, p0, p1):
    # check free indices
    for i in range( len(p0.tensorC.dimensions)):
      size0 = p0.tensorC.dimensions[i].size
      size1 = p1.tensorC.dimensions[i].size
      if size0 - size1 > 1:
        return 1
      elif size0 - size1 < -1:
        return -1
    # check summation indices
    for i in range(0, len(p0.operation.indexAssignmentsA)):
      index = p0.operation.indexAssignmentsA[i]
      inC = index < len(p0.tensorC.dimensions)
      inB = index in p0.operation.indexAssignmentsB
      if inB and not inC: # is summation dimension
        size0 = p0.tensorA.dimensions[i].size
        size1 = p1.tensorA.dimensions[i].size
        if size0 - size1 > 1:
          return 1
        elif size0 - size1 < -1:
          return -1
    return 0


  def getPSPsForSize( self, psps, sizeP):
    s = []
    for psp in psps:
      if self.compareSize(psp[0], sizeP) == 0:
        s.append(psp)
    return s

  def getPSPsFasterThan(self, psps, fasterThan):
    fastProblem = fasterThan[0]
    fastSolution = fasterThan[1]
    fastTime = fasterThan[2]
    fastSize = self.getSize(fastProblem)
    fastGFlops = self.getGFlops(fastProblem, fastTime)
    s = []
    for psp in psps:
      if self.getGFlops(psp[0], psp[2]) > fastGFlops:
        s.append(psp)
    return s

  def ruleToString(self, rule):
    # size
    # solution[0] + speed
    # solution[n] + speed
    # fallback + speed
    s = ""
    unorderedGroups = rule[0]
    fallback = rule[1]
    sizeMNK = self.getSize(fallback[0])**(1.0/3.0)
    s += "size = " + str(sizeMNK) + "^3\n"
    for group in unorderedGroups:
      for psp in group:
        s += "  %6.1f - %s___%s;" % ( self.getGFlops(psp[0], psp[2]), str(psp[0]), self.solutionWriter.getName(psp[1]) )
      s += "\n"
      
    s += "  %6.1f + %s___%s\n" % ( self.getGFlops(fallback[0], fallback[2]), str(fallback[0]), self.solutionWriter.getName(fallback[1]) )
    return s
    
  def getIndexOfNextLargestSize(self, psps, sizeP):
    for i in range(0, len(psps)):
      currentSizeP = psps[i][0]
      if self.compareSize(currentSizeP, sizeP) < 0:
        return i
    return len(psps)

  def removePSPsLargerOrEqual(self, psps, sizeP ):
    for psp in psps:
      if self.compareSize(psp[0], sizeP) > -1:
        psps.remove(psp)

  # two rules conflict only if they same elements in different order
  def rulesConflict(self, rule, newRule):
    unorderedGroups = rule[0]
    unorderedGroupsNew = newRule[0]
    for i in range(0,len(unorderedGroups)):
      unorderedGroup = unorderedGroups[i]
      for j in range(i+1, len(unorderedGroups)):
        unorderedGroupSubsequent = unorderedGroups[j]
        #rule says all elements in ordered unit are faster than all elements in subsequent ordered unit
        for iNew in range(0,len(unorderedGroupsNew)):
          unorderedGroupNew = unorderedGroupsNew[iNew]
          for jNew in range(iNew+1, len(unorderedGroupsNew)):
            unorderedGroupNewSubsequent = unorderedGroupsNew[jNew]
            #new rule says all elements in ordered unit are faster than all elements in subsequent ordered unit
            for psp in unorderedGroup:
              solution = psp[1]
              solutionInNewRuleSubsequent = False
              for pspNew in unorderedGroupNewSubsequent:
                if pspNew[1] == solution:
                  solutionInNewRuleSubsequent = True
                  break
              if solutionInNewRuleSubsequent:
                for pspSubsequent in unorderedGroupSubsequent:
                  solutionSubsequent = pspSubsequent[1]
                  solutionSubsequentInNewRule = False # if stile in ordered unit in new rule
                  for pspNew in unorderedGroupNew:
                    if pspNew[1] == solutionSubsequent:
                      solutionSubsequentInNewRule = True
                      break
                  if solutionSubsequentInNewRule:
                    # TODO does the conflict surpass tolerance?
                    print "rule conflict detected"
                    return True
    return False


  def mergeRules(self, rule, newRule):
    # already determined no conflicts
    # order only determined when multiple tiles show up in same problem size
    ugs = copy.deepcopy(rule[0]) # unordered groups
    nugs = copy.deepcopy(newRule[0]) # new unordered groups
    mugs = [] # merged unordered groups

    ugi = 0 # unordered group idx
    nugi = 0 # new unordered group idx
    #mugi = 0 # merged unordered group idx

    while ugi < len(ugs) or nugi < len(nugs):
      ugsValid = []
      if ugi < len(ugs):
        ugsValid.append(ugs[ugi])
        # eliminate ones in nugs[nugi+1+]
        for ug in ugsValid:
          invalid = False
          for j in range( nugi+1, len(nugs)):
            nug = nugs[j]
            for psp in nug:
              if psp[1] == ug[1]:
                # remove this from ugsValid
                invalid = True
          if invalid:
            ugsValid.remove(ug)

      nugsValid = []
      if nugi < len(nugs):
        nugsValid.append(nugs[nugi])
        # eliminate ones in ugs[ugi+1+]
        for nug in nugsValid:
          invalid = False
          for j in range( ugi+1, len(ugs)):
            ug = ugs[j]
            for psp in ug:
              if psp[1] == nug[1]:
                # remove this from nugsValid
                invalid = True
          if invalid:
            nugsValid.remove(ug)

      mugsValid = []
      for psp in ugsValid:
        mugsValid.append(psp)
      for psp in nugsValid:
        mugsValid.append(psp)
      mugs.append(mugsValid)
      
      if ugi < len(ugs):
        # remove ugsValid from ugs[ugi]
        for ug in ugsValid:
          ugs.remove(ug)
        if len(ugs[ugi]) == 0:
          ugi+=1
      if nugi < len(nugs):
        # remove nugsValid from nugs[nugi]
        for nug in nugsValid:
          nugs.remove(nug)
        if len(nugs[nugi]) == 0:
          nugi+=1

    # update rule with new unorderedGroups
    rule[0] = mugs
    return

  #############################################################################
  # write exact match level getSolution
  # chooses amongst sizes and mods
  #############################################################################
  def writeGetSolutionForExactMatch(self, exactMatch, inputProblemSolutionPairs):
    problemSolutionPairsUnsorted = copy.deepcopy(inputProblemSolutionPairs)
    problemSolutionPairs = self.sortSizePSPs(problemSolutionPairsUnsorted)
    print "total PSPs = " + str(len(problemSolutionPairs))
    # sort psps by descending size
    functionName = "getSolution_" + exactMatch.libString()

    s = ""
    
    s += "#include \"Problem.h\"\n"
    s += "#include \"CobaltGetSolution_" + exactMatch.libString() + ".h\"\n"
    for i in range(0, len(problemSolutionPairs)):
      problem = problemSolutionPairs[i][0]
      solution = problemSolutionPairs[i][1]
      time = problemSolutionPairs[i][2] # milliseconds
      s += "#include \"" + self.solutionWriter.getName(solution) + ".h\"\n"
      self.isSingleton(problem, solution)
      self.isFallback(problem, solution)
    s += "\n"
    s += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status ) {\n"




    # (a) determine fastest fallback psp at largest size
    ruleSizeThresholdUpperP = None
    largestSizeP = self.getLargestSize(problemSolutionPairs)
    pspsForLargestSize = self.getPSPsForSize(problemSolutionPairs, largestSizeP)
    fallbacksForLargestSize = self.getFallbacks(pspsForLargestSize)
    while len(fallbacksForLargestSize) < 1:
      indexOfNextLargestSize = self.getIndexOfNextLargestSize(problemSolutionPairs, largestSizeP)
      if indexOfNextLargestSize == len(problemSolutionPairs):
        # no next smallest size
        # no fallbacks
        break
      nextLargestSizeP = problemSolutionPairs[indexOfNextLargestSize]
      pspsForNextLargestSize = self.getPSPsForSize(problemSolutionPairs, nextLargestSizeP)
      fallbacksForLargestSize = self.getFallbacks(pspsForNextLargestSize)

    # if no fallbacks benchmarked, pick any and make its time slowest
    fallbackExists = len(fallbacksForLargestSize) > 0
    if not fallbackExists:
      fallbacksForLargestSize.append(problemSolutionPairs[0])
      fallbacksForLargestSize[0][2] = 1e10
    indexOfFastestFallback = self.getIndexOfFastest( fallbacksForLargestSize )
    fallback = fallbacksForLargestSize[indexOfFastestFallback]
    fallbackProblem = fallback[0]
    fallbackSolution = fallback[1]
    fallbackTime = fallback[2]
    fallbackGFlops = self.getGFlops(fallbackProblem, fallbackTime)

    # (b) going from largest problem to smallest problem, find smallest size for which this the speed of this fastest fallback solution at the problem size is still the fastest fallback solution
    if fallbackExists:
      fallbacks = self.getFallbacks(problemSolutionPairs)
      fallbackSizeThreshold = self.getSize(fallbackProblem)
      for i in range(0, len(fallbacks)):
        currentSize = self.getSize(fallbacks[i][0])
        if currentSize < fallbackSizeThreshold:
          # get speed of original fallback
          fallbacksForSize = self.getPSPsWithSize( fallbacks, currentSize)
          indexOfFallbackForSize = self.getIndexOfSolution(fallbacksForSize, fallbackSolution)
          if indexOfFallbackForSize >= len(fallbacksForSize):
            # fallback wasn't tested at this size
            continue
          fallbackProblemForSize = fallbacksForSize[indexOfFallbackForSize][0]
          fallbackSolutionForSize = fallbacksForSize[indexOfFallbackForSize][1]
          fallbackTimeForSize = fallbacksForSize[indexOfFallbackForSize][2]
          fallbackGFlopsForSize = self.getGFlops( fallbackProblemForSize, fallbackTimeForSize)
          # get speed of current fastest
          indexOfFastestFallbackForSize = self.getIndexOfFastest(fallbacksForSize)
          currentProblem = fallbacksForSize[indexOfFastestFallbackForSize][0]
          currentSolution = fallbacksForSize[indexOfFastestFallbackForSize][1]
          currentTime = fallbacksForSize[indexOfFastestFallbackForSize][2]
          currentGFlops = self.getGFlops(currentProblem, currentTime)
          if currentSolution != fallbackSolutionForSize:
            if currentGFlops > fallbackGFlopsForSize*self.tolerance:
              # starting with current size, there's a new fastest fallback
              break
            else:
              # new fallback is faster but still within tolerance
              fallbackSizeThreshold = currentSize
              continue
          else:
            # fallback is fastest at this size also
            fallbackSizeThreshold = currentSize
            continue
        else:
          continue # to to find smaller size
    else:
      # no fallback, so it's "valid" all the way down to zero
      fallbackSizeThreshold = 0
    # this size is the first fallback-threshold
    fallbackSizeThresholdMNK = fallbackSizeThreshold ** (1.0/3.0)
    print "Fallback " + str(fallbackSolution) + " is fastest down to size = " + str(fallbackSizeThreshold) + "(" + str(fallbackSizeThresholdMNK) + "^3)"
    
    # (c) at the largest size make list of all psps which are faster than the fallback; logically all must be branch.multiple AND exact match (else would be fallback)
    for psp in pspsForLargestSize:
      print "  %6.1f ~ %s___%s" % ( self.getGFlops(psp[0], psp[2]), str(psp[0]), self.solutionWriter.getName(psp[1]) )
    pspsFasterThanFallbackUnsorted = self.getPSPsFasterThan(pspsForLargestSize, fallback)

    # (d) sort list of fast psps in order of fastest to slowest
    pspsFasterThanFallback = self.sortSpeedPSPs(pspsFasterThanFallbackUnsorted)

    # (e) (a) and (c) are the "current rule" but without minimizing the size limit
    ruleSizeThresholdLowerP = fallback[0]
    unorderedGroups = []
    for psp in pspsFasterThanFallback:
      unorderedGroup = []
      unorderedGroup.append( psp )
      unorderedGroups.append( unorderedGroup )
    rule = [unorderedGroups, fallback, ruleSizeThresholdUpperP, ruleSizeThresholdLowerP]
    ruleString = self.ruleToString(rule)
    print ruleString

    if len(pspsFasterThanFallback) > 0:
      # find size threshold of rule
      # (f) incrementally move down in size to fallback-threshold, at each size make sorted list of all psps faster than fallback
      indexOfNextLargestSize = self.getIndexOfNextLargestSize(problemSolutionPairs, fallback[0])
      nextLargestSizeP = problemSolutionPairs[indexOfNextLargestSize]
      while self.compareSize(nextLargestSizeP, fallback[0]) < 0:
        pspsForCurrentSize = self.getPSPsForSize(problemSolutionPairs, nextLargestSizeP)
        pspsFasterThanFallbackCurrentSizeUnsorted = self.getPSPsFasterThan(pspsForCurrentSizeSize, fallback)
        pspsFasterThanFallbackCurrentSize = self.sortSpeedPSPs(pspsFasterThanFallbackCurrentSizeUnsorted)
        unorderedGroups = []
        for psp in pspsFasterThanFallbackCurrentSize:
          unorderedGroup = []
          unorderedGroup.append( psp )
          unorderedGroups.append( unorderedGroup )
        newRule = [unorderedGroups, fallback, ruleSizeThresholdUpperP, nextLargestSizeP]

        if self.rulesConflict(rule, newRule):
          # current rule is "the rule" with correct size threshold and correct
          break
        else:
          # we can make new rule which is at smaller size then "the rule" and may add more tiles without losing performance
          self.mergeRules(rule, newRule)
        
        indexOfNextLargestSize = self.getIndexOfNextLargestSize(problemSolutionPairs, nextLargestSizeP)
        nextLargestSizeP = problemSolutionPairs[indexOfNextLargestSize]

    # (g) if (f) conflicts with (e) by more than tolerance, then this is the size threshold for rule
    # repeat (e) and (g)
    else:
      # fallback is size threshold
      rule.append(fallbackProblem)

    # (h) remove psps which have size greater than rule-threshold
    self.removePSPsLargerOrEqual(problemSolutionPairs, rule[2])

    # (i) begin again at (a)

    

    s += "  return nullptr;\n"


    s += "}\n"
    s += "\n"


    # header file
    h = ""
    h += "#ifndef COBALT_" + functionName.upper() + "_H\n"
    h += "#define COBALT_" + functionName.upper() + "_H\n"
    h += "\n"
    h += "#include \"Cobalt.h\"\n"
    h += "\n"
    h += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"
    return (s, h)

  
  #############################################################################
  # write cmake file for CobaltLib solution selection
  #############################################################################
  def writeCobaltLibCMake(self, subdirectory):
    s = "# CobaltLib.cmake\n"
    s += "\n"
    s += "include( ${CobaltLib_KernelFiles_CMAKE_DYNAMIC} )\n"
    s += "include( ${CobaltLib_SolutionFiles_CMAKE_DYNAMIC} )\n"
    s += "\n"
    s += "set( CobaltLib_SRC_GENERATED_DYNAMIC\n"
    
    for deviceProfile, exactMatches in self.psMap.iteritems():
      print str(deviceProfile), str(exactMatches)
      # (2) Write Device-Level Solution Selection files
      baseName = "CobaltGetSolution_" + deviceProfile.libString()
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".cpp\n"
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".h\n"

      for exactMatch, problems in exactMatches.iteritems():
        baseName = "CobaltGetSolution_" + exactMatch.libString()
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".cpp\n"
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".h\n"
    s += ")\n"
    s += "\n"
    s += "source_group(CobaltGen\\\\Backend FILES\n"
    s += "  ${CobaltLib_SRC_GENERATED_STATIC}\n"
    s += "  ${CobaltLib_SRC_GENERATED_DYNAMIC} )\n"
    s += "\n"
    return s