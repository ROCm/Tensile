//#include "LibraryClient.h"
#include "Tensile.h"
#include "GeneratedHeader.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <iomanip>

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {
  std::cout << "usage: LibraryClient functionIdx sizes" << std::endl;
  for (unsigned int i = 0; i < numFunctions; i++) {
    std::cout << "(" << i << ") " << functionNames[i] << std::endl;
  }
  return 0;
}
