CLFLAGS =       -arch=sm_52 -O3 -lm

CUDA_PATH = /usr/local/cuda-7.5
NVCC = $(CUDA_PATH)/bin/nvcc
GDB = $(CUDA_PATH)/bin/cuda-gdb

bk: bk.cu
	$(NVCC) -o bk.exe bk.cu $(CLFLAGS)

deb: bk.cu
	$(NVCC) -g -G bk.cu $(CLFLAGS)

do: bk.exe
	./bk.exe 30367.inp

run: bk.exe
	./bk.exe 554763.inp

smp: bk.exe
	./bk.exe sample.inp

gdb: a.out
	$(GDB) a.out
db: a.out
	gdb a.out

nvvp:
	/usr/local/cuda-7.5/bin/nvvp

clean:
	rm -f bk.exe
	rm -f a.out
