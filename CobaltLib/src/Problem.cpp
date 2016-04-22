#include "Problem.h"
#include "Tools.h"
#include "StructOperations.h"

#include <algorithm>
#include <functional>
#include <assert.h>

namespace Cobalt {

/*******************************************************************************
 * constructor
 * - sorts indices so that different problems (constructor arguments)
 * which map to same solution appear as same problem (object state)
 ******************************************************************************/
Problem::Problem(
    CobaltTensor inputTensorC,
    CobaltTensor inputTensorA,
    CobaltTensor inputTensorB,
    unsigned int *inputIndexAssignmentsA,
    unsigned int *inputIndexAssignmentsB,
    CobaltOperationType inputOperationType,
    CobaltDataType inputAlphaType,
    CobaltDataType inputBetaType,
    bool inputUseOffsets,
    CobaltDeviceProfile inputDeviceProfile ) :
  tensorC( inputTensorC ),
  tensorA( inputTensorA ),
  tensorB( inputTensorB ),
  operationType(inputOperationType),
  alphaType( inputAlphaType ),
  betaType( inputBetaType ),
  useOffsets(inputUseOffsets),
  deviceProfile( inputDeviceProfile ),
  indicesA(inputIndexAssignmentsA, inputIndexAssignmentsA + inputTensorA.numDimensions),
  indicesB(inputIndexAssignmentsB, inputIndexAssignmentsB + inputTensorB.numDimensions)
{

  // determine batch, free and summation indices
  //indicesSummation.clear();
  for (unsigned int i = 0; i < tensorC.numDims() + tensorA.numDims(); i++) {
    bool inC = i < tensorC.numDims();
    unsigned int idxA = static_cast<unsigned int>(std::find(indicesA.begin(), indicesA.end(), i) - indicesA.begin());
    unsigned int idxB = static_cast<unsigned int>(std::find(indicesB.begin(), indicesB.end(), i) - indicesB.begin());
    bool inA = idxA < indicesA.size();
    bool inB = idxB < indicesB.size();

    // batch index
    if (inC && (inA && inB)) {
      indicesBatch.push_back(i);

      // free index
    } else if (inC && (inA || inB)) {
      indicesFree.push_back(i);

      // ERROR - unused free index
    } else if (inC && !inA && !inB) {
      throw cobaltStatusOperationIndexUnassigned;

      // summation index
    } else if (!inC && inA && inB) {
      indicesSummation.push_back(std::make_pair(idxA, idxB));

      // ERROR - index mismatch
    } else if (!inC && (inA || inB)) {
      throw cobaltStatusOperationSummationIndexAssignmentsInvalid;

      // this is okay, we just iterated over too many indices
    } else if (!inC && !inA && !inB) {

      // are there any other ERRORS I haven't thought of?
    } else {
      printf("Cobalt::Problem::constructor() - Error; mismatch I hadn't thought of.\n");
      throw cobaltStatusProblemNotFound;
    }
  }
  //printf("%s\n", toStringXML(2).c_str());
}

bool Problem::sortIndicesC( unsigned int i, unsigned int j) const {
  return tensorC[i].stride > tensorC[j].stride;
}
bool Problem::sortSummationIndexDescending( std::pair<unsigned int, unsigned int> i, std::pair<unsigned int, unsigned int> j) const {
  unsigned int iStrideA = tensorA[i.first].stride;
  unsigned int iStrideB = tensorB[i.second].stride;
  unsigned int jStrideA = tensorA[j.first].stride;
  unsigned int jStrideB = tensorB[j.second].stride;
  return iStrideA+iStrideB > jStrideA + jStrideB;
}

/*******************************************************************************
 * validate
 ******************************************************************************/
CobaltStatus Problem::validate( ) {

  /* tensorA,B */
  if (tensorA.numDims() != tensorB.numDims()) {
    assert(false);
    return cobaltStatusOperandNumDimensionsMismatch;
  }


  /* operation */
  // every element must correspond to a valid free idx or valid sum idx
  // no duplicates
  if (indicesFree.size()%2 != 0
      || indicesFree.size() < 2) {
    assert(false);
    return cobaltStatusOperationNumFreeIndicesInvalid;
  }
  if (indicesFree.size()/2
      + indicesBatch.size()
      + indicesSummation.size()
      != tensorA.numDims() ) {
    assert(false);
    return cobaltStatusOperationOperandNumIndicesMismatch;
  }
  if (indicesFree.size() + indicesBatch.size()
      != tensorC.numDims() ) {
    assert(false);
    return cobaltStatusOperationNumFreeIndicesInvalid;
  }
  if (indicesSummation.size() < 1 ) {
    assert(false);
    return cobaltStatusOperationNumSummationIndicesInvalid;
  }
  size_t maxAssignmentIndex = indicesFree.size() + indicesBatch.size() + indicesSummation.size() - 1;
  for (size_t i = 0; i < tensorA.numDims(); i++) {
    if (indicesA[i] > maxAssignmentIndex) {
      assert(false);
      return cobaltStatusOperationIndexAssignmentInvalidA;
    }
    if (indicesB[i] > maxAssignmentIndex) {
      assert(false);
      return cobaltStatusOperationIndexAssignmentInvalidB;
    }
    for (size_t j = i+1; j < tensorA.numDims(); j++) {
      if ( indicesA[i] == indicesA[j] ) {
        assert(false);
        return cobaltStatusOperationIndexAssignmentDuplicateA;
      }
      if ( indicesB[i] == indicesB[j] ) {
        assert(false);
        return cobaltStatusOperationIndexAssignmentDuplicateB;
      }
    }
  }

  /* indexAssignments */
  // matching indices must have same size
  for (size_t i = 0; i < indicesA.size(); i++) {
    unsigned int indexAssignment = indicesA[i];
    if (indexAssignment < tensorC.numDims()) { // match C
      if (tensorC[indexAssignment].size != tensorA[i].size) {
        assert(false);
        return cobaltStatusOperationIndexAssignmentInvalidA;
      }
    } else { // match B
      // find this index in B
      bool indexFound = false;
      unsigned int indexB;
      for (unsigned int j = 0; j < tensorB.numDims(); j++) {
        if (indicesB[j] == indexAssignment) {
          indexFound = true;
          indexB = j;
          break;
        }
      }
      if (indexFound) {
        if (tensorB[indexB].size != tensorA[i].size) {
          assert(false);
          return cobaltStatusOperationIndexAssignmentInvalidA;
        }
      } else {
        assert(false);
        return cobaltStatusOperationIndexUnassigned;
      }
    }
  }
  for (size_t i = 0; i < indicesB.size(); i++) {
    unsigned int indexAssignment = indicesB[i];
    if (indexAssignment < tensorC.numDims()) { // match C
      if (tensorC[indexAssignment].size != tensorB[i].size) {
        assert(false);
        return cobaltStatusOperationIndexAssignmentInvalidB;
      }
    } else { // match A
      // find this index in A
      bool indexFound = false;
      unsigned int indexA;
      for (unsigned int j = 0; j < tensorA.numDims(); j++) {
        if (indicesA[j] == indexAssignment) {
          indexFound = true;
          indexA = j;
          break;
        }
      }
      if (indexFound) {
        if (tensorA[indexA].size != tensorB[i].size) {
          assert(false);
          return cobaltStatusOperationIndexAssignmentInvalidB;
        }
      } else {
        assert(false);
        return cobaltStatusOperationIndexUnassigned;
      }
    }
  }

  return cobaltStatusSuccess;
}


/*******************************************************************************
 * toString
 ******************************************************************************/
 std::string Problem::toString() const {
   // assumes problem has already been validated
  std::string state = "";
  static const char *indexChars = "ijklmnopqrstuvwxyz";

  state += "C[";
  state += indexChars[0];
  state += ":";
  state += std::to_string(tensorC.dimensions[0].size);
  for (size_t i = 1; i < tensorC.numDims(); i++) {
    state += ",";
    state += indexChars[i];
    state += ":";
    state += std::to_string(tensorC.dimensions[i].size);
  }
  state += "] = Sum(";
  state += indexChars[tensorC.numDims()];
  state += ":";
  for (size_t j = 0; j < tensorA.numDims(); j++) {
    if (indicesA[j] == 0 + tensorC.numDims()) {
      state += std::to_string(tensorA.dimensions[j].size);
    }
  }
  //state += std::to_string(boundIndexSizes[0]);
  for (size_t i = 1; i < indicesSummation.size(); i++) {
    state += ",";
    state += indexChars[tensorA.numDims()+i];
    state += ":";
    for (size_t j = 0; j < tensorA.numDims(); j++) {
      if (indicesA[j] == i+tensorC.numDims()) {
        state += std::to_string(tensorA.dimensions[j].size);
      }
    }
  }
  state += ") A[";
  for (size_t i = 0; i < tensorA.numDims(); i++) {
    state += indexChars[indicesA[i]];
    if (i < tensorA.numDims()-1) {
      state += ",";
    }
  }
  state += "] * B[";
  for (size_t i = 0; i < tensorB.numDims(); i++) {
    state += indexChars[indicesB[i]];
    if (i < tensorB.numDims()-1) {
      state += ",";
    }
  }
  state += "]";
  return state;
} // toString

/*******************************************************************************
 * toStringXML
 ******************************************************************************/
std::string Problem::toStringXML( size_t indentLevel ) const {
  std::string state = Cobalt::indent(indentLevel);
  state += "<P>\n";
  state += tensorC.toStringXML( indentLevel+1, "C");
  state += tensorA.toStringXML( indentLevel+1, "A");
  state += tensorB.toStringXML( indentLevel+1, "B");
  state += toStringOperationXML( indentLevel+1);
  state += deviceProfile.toStringXML( indentLevel+1);
  state += Cobalt::indent(indentLevel) + "</P>\n";
  return state;
}

std::string Problem::toStringOperationXML( size_t indentLevel ) const {
  std::string state = Cobalt::indent(indentLevel);
  state += "<O";
  state += " t=\"" + std::to_string(operationType) + "\"";
  state += " a=\""+std::to_string(alphaType)+"\"";
  state += " b=\""+std::to_string(betaType)+"\"";
  state += " o=\""+std::to_string(useOffsets)+"\"";
  state += " nF=\""+std::to_string(indicesFree.size())+"\"";
  state += " nB=\""+std::to_string(indicesBatch.size())+"\"";
  state += " nS=\""+std::to_string(indicesSummation.size())+"\"";
  state += " >\n";
  // type
  // index assignments A
  state += Cobalt::indent(indentLevel+1) + "<IA";
  state += " n=\"" + std::to_string(indicesA.size()) + "\"";
  for (size_t i = 0; i < indicesA.size(); i++) {
    state += " i" + std::to_string(i) + "=\"" + std::to_string(indicesA[i]) + "\"";
  }
  state += " />\n";
  // index assignments B
  state += Cobalt::indent(indentLevel+1) + "<IB";
  state += " n=\"" + std::to_string(indicesB.size()) + "\"";
  for (size_t i = 0; i < indicesB.size(); i++) {
    state += " i" + std::to_string(i) + "=\"" + std::to_string(indicesB[i]) + "\"";
  }
  state += " />\n";

  state += Cobalt::indent(indentLevel) + "</O>\n";
  return state;
}

/*******************************************************************************
 * accessors
 ******************************************************************************/
CobaltDataType Problem::getDataTypeC() const { return tensorC.dataType; }
CobaltDataType Problem::getDataTypeA() const { return tensorA.dataType; }
CobaltDataType Problem::getDataTypeB() const { return tensorB.dataType; }
CobaltDataType Problem::getDataTypeAlpha() const { return alphaType; }
CobaltDataType Problem::getDataTypeBeta() const { return betaType; }
bool Problem::useAlpha() const { return alphaType != cobaltDataTypeNone; }
bool Problem::useBeta() const { return betaType != cobaltDataTypeNone; }
size_t Problem::alphaSize() const { return sizeOf(alphaType); }
size_t Problem::betaSize() const { return sizeOf(betaType); }
bool Problem::deviceIsReference() const {
  return strcmp( deviceProfile.devices[0].name.c_str(), "cpu" ) == 0;
}

/*******************************************************************************
 * comparator
 ******************************************************************************/
bool Problem::operator<(const Problem & other) const {
  
  // tensor C
  if( tensorC < other.tensorC) {
    return true;
  } else if ( other.tensorC < tensorC ) {
    return false;
  }

  // tensor A
  if( tensorA < other.tensorA) {
    return true;
  } else if (other.tensorA < tensorA ) {
    return false;
  }

  // tensor B
  if( tensorB < other.tensorB) {
    return true;
  } else if ( other.tensorB < tensorB ) {
    return false;
  }

  // type
  if (operationType < other.operationType) {
    return true;
  } else if (other.operationType < operationType) {
    return false;
  }
  if (alphaType < other.alphaType) {
    return true;
  } else if (other.alphaType < alphaType) {
    return false;
  }
  if (betaType < other.betaType) {
    return true;
  } else if (other.betaType < betaType) {
    return false;
  }

  // index assignments
  if (indicesFree < other.indicesFree) {
    return true;
  } else if (other.indicesFree < indicesFree) {
    return false;
  }
  if (indicesBatch < other.indicesBatch) {
    return true;
  } else if (other.indicesBatch < indicesBatch) {
    return false;
  }
  if (indicesSummation < other.indicesSummation) {
    return true;
  } else if (other.indicesSummation < indicesSummation) {
    return false;
  }
  if (indicesA < other.indicesA) {
    return true;
  } else if (other.indicesA < indicesA) {
    return false;
  }
  if (indicesB < other.indicesB) {
    return true;
  } else if (other.indicesB < indicesB) {
    return false;
  }
  // device
  if( deviceProfile < other.deviceProfile) {
    return true;
  } else if ( other.deviceProfile < deviceProfile ) {
    return false;
  }

  // identical
  return false;
}

} // namespace