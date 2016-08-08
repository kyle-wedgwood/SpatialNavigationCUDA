NVCC = nvcc
RM = rm -f
CUFLAGS  = -x cu -arch=compute_30 -code=sm_30
CPPFLAGS = -x c++ --std=c++11 -arch=compute_30 -code=sm_30
LDFLAGS = -arch=compute_30 -code=sm_30
CUINC  = -I. -dc -I/usr/local/include
INC = -I. -c -I/usr/local/include

OBJS = mainNaive.o

all: $(OBJS)
		$(NVCC) $(LDFLAGS) $(OBJS) -o Driver

%.o: %.cu
		$(NVCC) $(CUFLAGS) $(CUINC) $< -o $@

%.o: %.cpp
		$(NVCC) $(CPPFLAGS) $(INC) $< -o $@

clean:
		rm -f *.o Driver
