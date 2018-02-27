/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#include <vector>
#include <iostream>


class TensorDims {
public:
  TensorDims(
    const std::string name,
    unsigned int numIndices,
    unsigned int firstSummationIndex,
    const unsigned int *indexedSizes,
    const unsigned int *indexAssignments);

  void print() const;

  size_t computeMemoryOffset(size_t elementIndex);

public:
  // Sizes for each dimension of the matrix
  // First size is the coalesced dimension with fastest-changing address
  std::vector<unsigned int> sizes;

  size_t totalSize;

  // strides in logical element dimension, these do not necessarily correspond to memory location 
  std::vector<unsigned int> elementStrides;

  // strides in memory, useful for calculating address
  std::vector<unsigned int> memoryStrides;

private:
  const std::string _name;
  unsigned int _numIndices;
  unsigned int _firstSummationIndex;
  std::vector<unsigned int> _indexAssignment;

};

TensorDims::TensorDims(
  const std::string name,
  unsigned int numIndices,
  unsigned int firstSummationIndex,
  const unsigned int *indexedSizes,
  const unsigned int *indexAssignments) :
    _name(name),
    _numIndices(numIndices),
    _firstSummationIndex(firstSummationIndex),
    sizes(numIndices,0),
    elementStrides(numIndices, 0),
    memoryStrides(numIndices, 0),
    _indexAssignment(numIndices, 0) {

  totalSize = 1;
  for (unsigned int i=0; i<numIndices; i++) {
    _indexAssignment[i] = indexAssignments[i];
    sizes[i] = indexedSizes[indexAssignments[i]];
    totalSize *= sizes[i];
  }

  for (unsigned int i=0; i<numIndices; i++) {
    elementStrides[i] = 1;
    for (unsigned int j=0; j<i; j++) {
      elementStrides[i] *= sizes[j];
    }
  }

  // Assume memoryStrides are same as elementStrides.
  // Changing this would require some modest changes to add a
  // YAML parm and pipe it through the client testbench.
  for (unsigned int i=0; i<numIndices; i++) {
    memoryStrides[i] = 1;
    for (unsigned int j=0; j<i; j++) {
      memoryStrides[i] *= sizes[j];
    }
  }
}


void TensorDims::print() const {

  std::cout << "Matrix:" << _name << "  indexAssignments:";
  for (unsigned int i=0; i<_numIndices; i++) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << _indexAssignment[i];
    if (_indexAssignment[i] >= _firstSummationIndex) 
        std::cout << "(sum)";
    else
        std::cout << "(free)";
  }
  std::cout << "\n";

  for (unsigned int i=0; i<_numIndices; i++) {
    std::cout << "  size[" << i << "]=" << sizes[i] 
              << (_indexAssignment[i] >= _firstSummationIndex ? " (sum)" : " (free)")
              << "\n";
  }
  for (unsigned int i=0; i<_numIndices; i++) {
    std::cout << "  elementStrides[" << i << "]="<<elementStrides[i]<<"\n";
  }
  for (unsigned int i=0; i<_numIndices; i++) {
    std::cout << "  memoryStrides[" << i << "]="<<memoryStrides[i]<<"\n";
  }
};


size_t TensorDims::computeMemoryOffset(size_t elementIndex) {

  size_t r = elementIndex;
  size_t offset = 0;
  for (int j=_numIndices-1; j>=0; j--) {
    offset += r / elementStrides[j] * memoryStrides[j];
    r = r % elementStrides[j];
  }

  return offset;
}


static const unsigned PrintElementIndex   = 0x1;
static const unsigned PrintElementOffset = 0x2;
static const unsigned PrintElementValue   = 0x4;


// Tensile_LIBRARY_PRINT_DEBUG  ??
#if 1
/*******************************************************************************
 * Print Tensor.
 * Elements from index[0] should appear on one row.
 * Matrix start "[" and stop "]" markers are printed at appropriate points,
 * when the indexing crosses a dimension boundary.  The markers are indented
 * to indicate which dimension is changing
 ******************************************************************************/
template< typename Type >
void printTensor(
    const std::string &name,
    const Type *data,
    unsigned int numIndices,
    unsigned int firstSummationIndex,
    const unsigned int *indexedSizes,
    const unsigned int *indexAssignments,
    //unsigned printMode=PrintElementIndex | PrintElementValue) {
    unsigned printMode=PrintElementValue) {

    TensorDims td(name, numIndices, firstSummationIndex, indexedSizes, indexAssignments);

    td.print();

    const unsigned int maxCols = 50;
    const unsigned int maxRows = 100;


    // Lower-numbered indices have more leading spaces
    // Highest index has zero and is aligned to left column
    std::vector<std::string> leadingSpaces(numIndices, "");
    for (unsigned int i=0; i<numIndices; i++) {
      for (unsigned int j=numIndices-1; j>i; j--) {
        leadingSpaces[i] += "  ";
      }
    }

    bool dbPrint = false;


#if 0
    if (maxCols != -1) {
        if (maxCols > 10) {
            colAfterElipse=5;
        } else {
            colAfterElipse=1;
        }
    }
#endif

    // Print the elements 
    for (size_t e=0; e<td.totalSize; e++) {

      // see if we are at an interesting boundary:
      for (int n=numIndices-1; n>=1; n--) {
        if (e % td.elementStrides[n] == 0) {
          // first element in this dimension:
          if (n==1) {
            // Label the row:
            std::cout << leadingSpaces[n];
            size_t r = e;
            for (int m=numIndices-1; m>=1; m--) {
              std::cout << r / td.elementStrides[m] ;
              r = r % td.elementStrides[m];
              if (m == 1) {
                std::cout << ",x : ";
              } else {
                std::cout << ",";
              }
            }
          } 
          std::cout << leadingSpaces[n] << "[ ";
          
          if (dbPrint) {
            unsigned int iter = e / td.elementStrides[n];
            std::cout << "// Start index=" << n 
                      <<  " iter=" << iter << " / " << td.elementStrides[n] << "\n"; 
          } 
          if (n>1) {
            std::cout << "\n";
          }
        }
      }

      // actually print the element:
      std::cout << "  ";
      size_t eo = td.computeMemoryOffset(e);
      if (printMode & PrintElementIndex) {
        std::cout  << e << ":";
      }
      if (printMode & PrintElementOffset) {
        std::cout << eo << ":";
      }
      if (printMode & PrintElementValue) {
        std::cout << data[eo];
      }

      for (int n=1; n<numIndices; n++) {
        if (e % td.elementStrides[n] == td.elementStrides[n]-1) {
          unsigned int iter = e / td.elementStrides[n];

          // last element in the index:
          std::cout << leadingSpaces[n]  << "]\n";
          if (dbPrint) {
            std::cout << "// End index=" << n 
                      << " iter=" << iter << " / " << td.elementStrides[n] << "\n";
          }
        } 
      }
    }
      

    std::cout << "\n";
}
#endif
