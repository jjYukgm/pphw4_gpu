CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++0x

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_20 -O3

OMPFLAGS                := -fopenmp
MPILIBS                 := -I/opt/mpich-3.2-gcc/include -L/opt/mpich-3.2-gcc/lib -lmpich

EXES                    := HW4_105061516_cuda.exe HW4_105061516_openmp.exe HW4_105061516_mpi.exe
EXES2                   := HW4_105061516_cuda2.exe HW4_105061516_cuda3.exe HW4_105061516_cuda4.exe HW4_105061516_openmp2.exe HW4_105061516_openmp3.exe HW4_105061516_openmp4.exe HW4_105061516_mpi2.exe HW4_105061516_mpi3.exe HW4_105061516_mpi4.exe

all: $(EXES)
all234: $(EXES2)


clean:
	rm -f $(EXES)

block_FW.exe: block_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

seq_FW.exe: seq_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

HW4_105061516_cuda.exe: HW4_105061516_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_105061516_openmp.exe: HW4_105061516_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_105061516_mpi.exe: HW4_105061516_mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?

HW4_105061516_cuda2.exe: HW4_105061516_cuda6_non.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_105061516_openmp2.exe: HW4_105061516_openmp6_non.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_105061516_mpi2.exe: HW4_105061516_mpi6_non.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?

HW4_105061516_cuda3.exe: HW4_105061516_cuda4_sm.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_105061516_openmp3.exe: HW4_105061516_openmp4_sm.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_105061516_mpi3.exe: HW4_105061516_mpi4_sm.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?

HW4_105061516_cuda4.exe: HW4_105061516_cuda5_st.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_105061516_openmp4.exe: HW4_105061516_openmp5_st.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_105061516_mpi4.exe: HW4_105061516_mpi5_st.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?

