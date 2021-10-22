CXX=/opt/rocm/bin/hipcc
LIBFLAGS=-L/opt/rocm/lib/
LIBS=-lamdhip64
INCFLAGS=-I/opt/rocm/include/

hipModuleLoadTiming.out: hipModuleLoadTiming.o
	$(CXX) -o $@ $(LIBFLAGS) $^
hipModuleLoadTiming.o: hipModuleLoadTiming.cpp
	$(CXX) -c $(INCFLAGS) $^
clean:
	rm *.o *.out
