#include "Problem.h"
#include "Tools.h"
#include "StructOperations.h"

#include <algorithm>

namespace Cobalt {

/*******************************************************************************
 * Tensor
 ******************************************************************************/
Tensor::Tensor(CobaltTensor tensor)
  : dataType(tensor.dataType),
  dimensions(tensor.dimensions, tensor.dimensions+tensor.numDimensions) { }

unsigned int Tensor::numDims() const {
  return dimensions.size();
}

std::string Tensor::toString() const {
  std::string state = "";
  return state;
}

std::string Tensor::toStringXML( size_t indent ) const {
  std::string state = Cobalt::indent(indent);
  state += "<Tensor numDimensions=\"" + std::to_string(dimensions.size())
      + "\"";
  state += " dataType=\"" + Cobalt::toString( dataType ) + "\"";
  state += " >\n";
  for (size_t i = 0; i < numDims(); i++) {
    state += Cobalt::indent(indent+1) + "<Dimension stride=\""
        + std::to_string(dimensions[i].stride) + "\"";
    state += " size=\"" + std::to_string(dimensions[i].size) + "\" />\n";
  }
  state += Cobalt::indent(indent) + "</Tensor>\n";
  return state;
}

/*******************************************************************************
 * DeviceProfile
 ******************************************************************************/
Device::Device( CobaltDevice device )
  : name(device.name),
  numComputeUnits(device.numComputeUnits),
  clockFrequency(device.clockFrequency) { }

void Device::init( CobaltDevice device ) {
  name.assign(device.name);
  numComputeUnits = device.numComputeUnits;
  clockFrequency = device.clockFrequency;
}
std::string toStringXML( const CobaltDevice device, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Device name=\"";
  state += device.name;
  state += "\"";
  state += " numComputeUnits=\"" + std::to_string(device.numComputeUnits) + "\"";
  state += " clockFrequency=\"" + std::to_string(device.clockFrequency) + "\"";
  state += " />\n";
  return state;
}

DeviceProfile::DeviceProfile( CobaltDeviceProfile profile)
  : devices(profile.numDevices) {
  for (unsigned int i = 0; i < profile.numDevices; i++) {
    devices[i].init(profile.devices[i]);
  }
}
std::string DeviceProfile::toStringXML( size_t indent ) const {
  std::string state = Cobalt::indent(indent);
  state += "<DeviceProfile";
  state += " numDevices=\"" + std::to_string(devices.size())
      + "\" >\n";
  for (size_t i = 0; i < devices.size(); i++) {
    state += devices[i].toStringXML(indent+1);
  }
  state += Cobalt::indent(indent) + "</DeviceProfile>\n";
  return state;
}

/*******************************************************************************
 * Problem
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
  invalidationCaughtInConstructor(false)
{

  
}

bool Problem::useAlpha() const {
  return alphaType != cobaltDataTypeNone;
}

bool Problem::useBeta() const {
  return betaType != cobaltDataTypeNone;
}



/*******************************************************************************
 * validate
 ******************************************************************************/
CobaltStatus Problem::validate( ) {

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

  
  for (unsigned int i = 0; i < tensorC.numDims() + tensorA.numDims(); i++) {
    bool inC = i < tensorC.numDims();
    unsigned int idxA = std::find( indicesA.begin(), indicesA.end(), i) - indicesA.begin();
    unsigned int idxB = std::find( indicesB.begin(), indicesB.end(), i) - indicesB.begin();
    bool inA = idxA < indicesA.size();
    bool inB = idxB < indicesB.size();

    // batch index
    if (inC && (inA && inB) ) {
      indicesBatch.push_back(i);

    // free index
    } else if (inC && (inA || inB) ) {
      indicesFree.push_back(i);

    // unused free index
    } else if (inC && !inA && !inB) {
      return cobaltStatusOperationIndexUnassigned;

    // summation index
    } else if (!inC && inA && inB) {
      indicesSummation.push_back( std::make_pair(idxA,idxB) );
      
      // index mismatch
    } else if (!inC && (inA || inB) ) {
      return cobaltStatusOperationSummationIndexAssignmentsInvalid;
      
      // this is okay, we just iterated over too many indices
    } else if (!inC && !inA && !inB) {

      // are there any other mismatches I haven't thought of?
    } else {
      printf("Cobalt::Problem::constructor() - Error; mismatch I hadn't thought of.\n");
      invalidationCaughtInConstructor = true;
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
      if ( indicesA[i]
          == indicesA[j] ) {
        return cobaltStatusOperationIndexAssignmentDuplicateA;
      }
          if ( indicesB[i]
          == indicesB[j] ) {
        return cobaltStatusOperationIndexAssignmentDuplicateB;
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
 * struct toString
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

CobaltDataType Problem::getDataTypeC() const { return tensorC.dataType; }
CobaltDataType Problem::getDataTypeA() const { return tensorA.dataType; }
CobaltDataType Problem::getDataTypeB() const { return tensorB.dataType; }
CobaltDataType Problem::getDataTypeAlpha() const { return alphaType; }
CobaltDataType Problem::getDataTypeBeta() const { return betaType; }


} // namespace