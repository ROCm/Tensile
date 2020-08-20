CXX=/opt/rocm/hip/bin/hipcc
LIBFLAGS=-L/opt/rocm/hip/lib/
LIBS=-lamdhip64
INCFLAGS=-I/opt/rocm/hip/include/

hipModuleLoadTiming.out: hipModuleLoadTiming.o
	$(CXX) -o $@ $(LIBFLAGS) $^
hipModuleLoadTiming.o: hipModuleLoadTiming.cpp
	$(CXX) -c $(INCFLAGS) $^
clean:
	rm *.o *.out
