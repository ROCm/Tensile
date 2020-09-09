#include <chrono>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>

using namespace std;
using namespace chrono;

/****************************
* Usage: ./hipModuleLoadTiming.out filename
*****************************/
int main(int argc, char** argv)
{

    if(argc < 2)
    {
        cout << "Usage: ./hipModuleLoadTiming.out filename" << std::endl;
        return 1;
    }

    const char* filename = argv[1];

    hipModule_t module;
    {
        time_point<system_clock> startTime = system_clock::now();
        hipModuleLoad(&module, filename);
        time_point<system_clock> endTime    = system_clock::now();
        duration<float>          difference = endTime - startTime;
        std::cout << "hipModuleLoad() took " << difference.count() << " seconds" << std::endl;
    }

    {
        time_point<system_clock> startTime = system_clock::now();
        std::ifstream            file(filename, std::ifstream::binary);
        if(file)
        {
            //Get length
            file.seekg(0, file.end);
            size_t length = file.tellg();
            file.seekg(0, file.beg);

            char* buffer = new char[length];

            file.read(buffer, length);
            time_point<system_clock> endTime    = system_clock::now();
            duration<float>          difference = endTime - startTime;
            std::cout << "Loading from filesystem to memory took " << difference.count()
                      << " seconds" << std::endl;
        }
        else
            cout << "File " << filename << " not found" << endl;
    }
}
