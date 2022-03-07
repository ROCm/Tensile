LDFLAGS = -lOpenCL 
INCLUDE = 

SRCS = $(wildcard *.cc) 

all: build_test 
build_test: $(SRCS)
	g++ -O3 $^ -o $@ $(INCLUDE) $(LDFLAGS) -I/opt/rocm/include/ 
clean:
	rm build_test 
