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
#include <iomanip>
#include <cstddef>

extern const char indexChars[];


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
  std::vector<unsigned int> _indexAssignments;

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
    _indexAssignments(numIndices, 0) {

  totalSize = 1;
  for (unsigned int i=0; i<numIndices; i++) {
    _indexAssignments[i] = indexAssignments[i];
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
  for (int i=_numIndices-1; i>=0; i--) {
    if (i != _numIndices-1) {
      std::cout << ", ";
    }
    std::cout << _indexAssignments[i];
    if (_indexAssignments[i] >= _firstSummationIndex) 
        std::cout << "(sum)";
    else
        std::cout << "(free)";
  }
  std::cout << "\n";

  for (int i=_numIndices-1; i>=0; i--) {
    std::cout << "  size[" << i << "]=" << sizes[i] 
              << (_indexAssignments[i] >= _firstSummationIndex ? " (sum)" : " (free)")
              << ",\'" << indexChars[_indexAssignments[i]] << "\'"
              << "\n";
  }
  for (int i=_numIndices-1; i>=0; i--) {
    std::cout << "  elementStrides[" << i << "]="<<elementStrides[i]<<"\n";
  }
  for (int i=_numIndices-1; i>=0; i--) {
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


static const unsigned PrintRowAddress       = 0x01;  // Print virtual address at beginning of row. If 0, print tensor coordinates in elements instead.
static const unsigned PrintRowElementCoord  = 0x02;  // Print tensor coordinates (using comma-separated element offsets in each dim) at beginning of each row

static const unsigned PrintElementIndex     = 0x04;  // Print index of element (0...n)
static const unsigned PrintElementOffset    = 0x08;  // Print offset of element, accounting for memory strides.  THis is an element offset not a byte offset.
static const unsigned PrintElementValue     = 0x10;  // Print value of element in its native type (typically half, float, or double)
static const unsigned PrintElementValueHex  = 0x20;  // Print hex value of element


// Tensile_LIBRARY_PRINT_DEBUG  ??
#if 1
/*******************************************************************************
 * Print Tensor.
 * Elements from index[0] should appear on one row.
 * Index[0] is the fastest moving and elements in the printed row are
 * adjacent in memory.
 * Matrix start "[" and stop "]" markers are printed at appropriate points,
 * when the indexing crosses a dimension boundary.  The markers are indented
 * to indicate which dimension is changing
 *****************************************************************************/
template< typename Type >
void printTensor(
    const std::string &name,
    const Type *data,
    unsigned int numIndices,
    unsigned int firstSummationIndex,
    const unsigned int *indexedSizes,
    const unsigned int *indexAssignments,
    //unsigned printMode=PrintRowAddress + PrintRowElementCoord + PrintElementIndex + PrintElementValue + PrintElementValueHex) {
    //unsigned printMode= PrintRowElementCoord + PrintElementIndex + PrintElementValue + PrintElementValueHex) {
    //unsigned printMode= PrintRowElementCoord + PrintElementIndex + PrintElementValue + PrintElementValueHex) {
    unsigned printMode=PrintRowElementCoord+PrintElementValueHex) {

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

    // Print order of elements:
    std::cout << "\n  ";
    for (int i=numIndices-1; i>=0; i--) {
        if (i != numIndices-1) {
          std::cout << ",";
        }
        std::cout << indexChars[indexAssignments[i]];
    }
    std::cout << "\n";

    // Print the elements 
    for (size_t e=0; e<td.totalSize; e++) {
      size_t eo = td.computeMemoryOffset(e);

      // see if we are at an interesting boundary:
      for (int n=numIndices-1; n>=1; n--) {
        if (e % td.elementStrides[n] == 0) {
          // first element in this dimension:
          if (n==1) {
            // Label the row:
            std::cout << leadingSpaces[n];
            size_t r = e;

            if (printMode & PrintRowAddress) {
              std::cout << &data[eo] << ":";
            } 
            if (printMode & PrintRowElementCoord) {
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

      // actually print the element, with leading ofset or address info:
      std::cout << "  ";
      if (printMode & PrintElementIndex) {
        std::cout  << e << ":";
      }
      if (printMode & PrintElementOffset) {
        std::cout << eo << ":";
      }
      if (printMode & PrintElementValue) {
        std::cout << data[eo];
      }
      if (printMode & PrintElementValueHex) {
        if (printMode & PrintElementValue) {
          std::cout << "/";
        }
        if (sizeof(Type) == 2) {
          std::cout << "0x" << std::setfill('0') << std::setw(4) << std::hex << *(uint16_t*)(&data[eo]) << std::dec;
        } else if (sizeof(Type) == 4) {
          std::cout << "0x" << std::setfill('0') << std::setw(8) << std::hex << *(uint32_t*)(&data[eo]) << std::dec;
        } else if (sizeof(Type) == 8) {
          std::cout << "0x" << std::setfill('0') << std::setw(16) << std::hex << *(uint64_t*)(&data[eo]) << std::dec;
        }
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
