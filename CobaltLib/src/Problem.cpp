#include "Problem.h"
#include "Tools.h"
#include "StructOperations.h"

#include <algorithm>
#include <functional>

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
    CobaltDeviceProfile inputDeviceProfile ) :
  tensorC( inputTensorC ),
  tensorA( inputTensorA ),
  tensorB( inputTensorB ),
  operationType(inputOperationType),
  alphaType( inputAlphaType ),
  betaType( inputBetaType ),
  deviceProfile( inputDeviceProfile ),
  indicesA(inputIndexAssignmentsA, inputIndexAssignmentsA + inputTensorA.numDimensions),
  indicesB(inputIndexAssignmentsB, inputIndexAssignmentsB + inputTensorB.numDimensions),
  constructorStatus(cobaltStatusSuccess)
{

  // (1) determine batch, free and summation indices
  for (unsigned int i = 0; i < tensorC.numDims() + tensorA.numDims(); i++) {
    bool inC = i < tensorC.numDims();
    unsigned int idxA = static_cast<unsigned int>(std::find( indicesA.begin(), indicesA.end(), i) - indicesA.begin());
    unsigned int idxB = static_cast<unsigned int>(std::find( indicesB.begin(), indicesB.end(), i) - indicesB.begin());
    bool inA = idxA < indicesA.size();
    bool inB = idxB < indicesB.size();

    if (!inC && inA && inB) {
      indicesSummation.push_back( std::make_pair(idxA,idxB) );
    }
  }
  //printf("Problem::(0)\n%s\n", toStringXML(2).c_str());

  // (2) sort C indices by decreasing stride
  std::vector<unsigned int> indicesNewC;
  for (unsigned int i = 0; i < tensorC.numDims(); i++) {
    indicesNewC.push_back(i);
  }
  std::sort(indicesNewC.begin(), indicesNewC.end(), std::bind(&Cobalt::Problem::sortIndicesC, this, std::placeholders::_1, std::placeholders::_2) );
  
  // (3) sort summation indices by decreasing total(a+b) stride
  std::vector<std::pair<unsigned int,unsigned int>> indicesSummationNew = indicesSummation;
  std::sort(indicesSummationNew.begin(), indicesSummationNew.end(), std::bind(&Cobalt::Problem::sortSummationIndexDescending, this, std::placeholders::_1, std::placeholders::_2) );

  // (4) based on (2) rearrange tensorC dimensions
  std::vector<CobaltDimension> dimensionsNewC(tensorC.numDims());
  for (unsigned int i = 0; i < tensorC.numDims(); i++) {
    dimensionsNewC[indicesNewC[i]] = tensorC[i];
  }
  tensorC.dimensions = dimensionsNewC;

  // (5) based on (2,3) rearrange indicesA,B
  std::vector<unsigned int> indicesNew = indicesNewC;
  // append indicesSummationSorted to indicesNew
  for (size_t i = 0; i < indicesSummation.size(); i++) {
    auto pair = indicesSummation[i];
    auto foundIdx = std::find(indicesSummationNew.begin(), indicesSummationNew.end(), pair);
    size_t idx = foundIdx - indicesSummationNew.begin();
    indicesNew.push_back(static_cast<unsigned int>(idx + indicesNewC.size()));
  }
  for (auto i = 0; i < indicesA.size(); i++) {
    indicesA[i] = indicesNew[indicesA[i]];
  }
  for (auto i = 0; i < indicesB.size(); i++) {
    indicesB[i] = indicesNew[indicesB[i]];
  }
  //printf("Problem::(5)\n%s\n", toStringXML(2).c_str());

  // (6) A gets indices largest->smallest stride
  std::vector<unsigned int> orderNewA = tensorA.sortDimensions();
  // (6a) A indexAssignments
  std::vector<unsigned int> indicesNewA;
  for (auto i = 0; i < indicesA.size(); i++) {
    indicesNewA.push_back( indicesA[orderNewA[i] ] );
  }
  indicesA = indicesNewA;
  //printf("Problem::(6)\n%s\n", toStringXML(2).c_str());
  

  // (7) B gets indices largest->smallest stride
  std::vector<unsigned int> orderNewB = tensorB.sortDimensions();
  // (7a) B indexAssignments
  std::vector<unsigned int> indicesNewB;
  for (auto i = 0; i < indicesB.size(); i++) {
    indicesNewB.push_back(indicesB[orderNewB[i]]);
  }
  indicesB = indicesNewB;
  //printf("Problem::(7)\n%s\n", toStringXML(2).c_str());

  // (8) determine batch, free and summation indices
  indicesSummation.clear();
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
    }
    else if (inC && (inA || inB)) {
      indicesFree.push_back(i);

      // ERROR - unused free index
    }
    else if (inC && !inA && !inB) {
      constructorStatus = cobaltStatusOperationIndexUnassigned;

      // summation index
    }
    else if (!inC && inA && inB) {
      indicesSummation.push_back(std::make_pair(idxA, idxB));

      // ERROR - index mismatch
    }
    else if (!inC && (inA || inB)) {
      constructorStatus = cobaltStatusOperationSummationIndexAssignmentsInvalid;

      // this is okay, we just iterated over too many indices
    }
    else if (!inC && !inA && !inB) {

      // are there any other ERRORS I haven't thought of?
    }
    else {
      printf("Cobalt::Problem::constructor() - Error; mismatch I hadn't thought of.\n");
      constructorStatus = cobaltStatusProblemNotFound;
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

  if (constructorStatus != cobaltStatusSuccess) {
    return cobaltStatusSuccess;
  }

  /* tensorA */
  if (tensorA.numDims() < 1
    || tensorA.numDims() > CobaltTensor::maxDimensions ) {
      return cobaltStatusTensorNumDimensionsInvalidA;
  }
  for (size_t i = 0; i < tensorA.numDims(); i++) {
    if (tensorA.dimensions[i].size < 1) {
      return cobaltStatusTensorDimensionSizeInvalidA;
    }
    if (tensorA.dimensions[i].stride < 1) {
      return cobaltStatusTensorDimensionStrideInvalidA;
    }
  }

  /* tensorB */
  if (tensorB.numDims() < 1
    || tensorB.numDims() > CobaltTensor::maxDimensions ) {
      return cobaltStatusTensorNumDimensionsInvalidB;
  }
  for (size_t i = 0; i < tensorB.numDims(); i++) {
    if (tensorB.dimensions[i].size < 1) {
      return cobaltStatusTensorDimensionSizeInvalidB;
    }
    if (tensorB.dimensions[i].stride < 1) {
      return cobaltStatusTensorDimensionStrideInvalidB;
    }
  }

  /* tensorA,B */
  if (tensorA.numDims() != tensorB.numDims()) {
    return cobaltStatusOperandNumDimensionsMismatch;
  }
  
  /* tensorC */
  if (tensorC.numDims() < 1
    || tensorC.numDims() > CobaltTensor::maxDimensions ) {
      return cobaltStatusTensorNumDimensionsInvalidC;
  }
  for (size_t i = 0; i < tensorC.numDims(); i++) {
    if (tensorC.dimensions[i].size < 1) {
      return cobaltStatusTensorDimensionSizeInvalidC;
    }
    if (tensorC.dimensions[i].stride < 1) {
      return cobaltStatusTensorDimensionStrideInvalidC;
    }
  }


  /* operation */
  // every element must correspond to a valid free idx or valid sum idx
  // no duplicates
  if (indicesFree.size()%2 != 0
      || indicesFree.size() < 2) {
    return cobaltStatusOperationNumFreeIndicesInvalid;
  }
  if (indicesFree.size()/2
      + indicesBatch.size()
      + indicesSummation.size()
      != tensorA.numDims() ) {
    return cobaltStatusOperationOperandNumIndicesMismatch;
  }
  if (indicesFree.size() + indicesBatch.size()
      != tensorC.numDims() ) {
    return cobaltStatusOperationNumFreeIndicesInvalid;
  }
  if (indicesSummation.size() < 1 ) {
    return cobaltStatusOperationNumSummationIndicesInvalid;
  }
  size_t maxAssignmentIndex = indicesFree.size() + indicesBatch.size() + indicesSummation.size() - 1;
  for (size_t i = 0; i < tensorA.numDims(); i++) {
    if (indicesA[i] > maxAssignmentIndex) {
      return cobaltStatusOperationIndexAssignmentInvalidA;
    }
    if (indicesB[i] > maxAssignmentIndex) {
      return cobaltStatusOperationIndexAssignmentInvalidB;
    }
    for (size_t j = i+1; j < tensorA.numDims(); j++) {
      if ( indicesA[i] == indicesA[j] ) {
        return cobaltStatusOperationIndexAssignmentDuplicateA;
      }
      if ( indicesB[i] == indicesB[j] ) {
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
        return cobaltStatusOperationIndexAssignmentInvalidA;
      }
    }
    else { // match B
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
          return cobaltStatusOperationIndexAssignmentInvalidA;
        }
      }
      else {
        return cobaltStatusOperationIndexUnassigned;
      }
    }
  }
  for (size_t i = 0; i < indicesB.size(); i++) {
    unsigned int indexAssignment = indicesB[i];
    if (indexAssignment < tensorC.numDims()) { // match C
      if (tensorC[indexAssignment].size != tensorB[i].size) {
        return cobaltStatusOperationIndexAssignmentInvalidB;
      }
    }
    else { // match A
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
          return cobaltStatusOperationIndexAssignmentInvalidB;
        }
      }
      else {
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
  //state += std::to_string(boundIndexSizes[0]);
  for (size_t i = 1; i < indicesSummation.size(); i++) {
    state += ",";
    state += indexChars[tensorA.numDims()+i];
    state += ":";
    for (size_t j = 0; j < tensorA.numDims(); j++) {
      if (indicesA[j] == i) {
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
  state += "<Problem string=\"" + toString() + "\">\n";
  state += tensorC.toStringXML( indentLevel+1);
  state += tensorA.toStringXML( indentLevel+1);
  state += tensorB.toStringXML( indentLevel+1);
  state += toStringOperationXML( indentLevel+1);
  state += deviceProfile.toStringXML( indentLevel+1);
  state += Cobalt::indent(indentLevel) + "</Problem>\n";
  return state;
}

std::string Problem::toStringOperationXML( size_t indentLevel ) const {
  std::string state = Cobalt::indent(indentLevel);
  state += "<Operation ";
  state += "useAlpha=\""+std::to_string(useAlpha())+"\" ";
  state += "alphaType=\""+std::to_string(alphaType)+"\" ";
  state += "useBeta=\""+std::to_string(useBeta())+"\" ";
  state += "betaType=\""+std::to_string(betaType)+"\" ";
  state += "numIndicesFree=\""+std::to_string(indicesFree.size())+"\" ";
  state += "numIndicesBatch=\""+std::to_string(indicesBatch.size())+"\" ";
  state += "numIndicesSummation=\""+std::to_string(indicesSummation.size())+"\" ";
  state += ">\n";
  state += Cobalt::indent(indentLevel+1);
  // type
  state += "<Type enum=\"" + std::to_string(operationType) + "\"";
  state += " string=\"" + Cobalt::toString(operationType) + "\" />\n";
  // operationIndexAssignmentsA
  state += Cobalt::indent(indentLevel+1) + "<IndexAssignments tensor=\"A\" >\n";
  for (size_t i = 0; i < indicesA.size(); i++) {
    state += Cobalt::indent(indentLevel+2);
    state += "<IndexAssignment";
    state += " index=\"" + std::to_string(i) + "\"";
    state += " indexAssignment=\""
        + std::to_string(indicesA[i]) + "\"";
    state += " />\n";
  }
  state += Cobalt::indent(indentLevel+1) + "</IndexAssignments>\n";
  // operationIndexAssignmentsB
  state += Cobalt::indent(indentLevel+1) + "<IndexAssignments tensor=\"B\" >\n";
  for (size_t i = 0; i < indicesB.size(); i++) {
    state += Cobalt::indent(indentLevel+2);
    state += "<IndexAssignment";
    state += " index=\"" + std::to_string(i) + "\"";
    state += " indexAssignment=\""
      + std::to_string(indicesB[i]) + "\"";
    state += " />\n";
  }
  state += Cobalt::indent(indentLevel+1) + "</IndexAssignments>\n";
  state += Cobalt::indent(indentLevel) + "</Operation>\n";
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