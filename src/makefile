EXE=main.cu
OBJ=main.seery

input_Matrix=/data/seery/cuda_learn/dataset/bcsstk16/bcsstk16.mtx

test_Matrix=/data/suitesparse_collection/af_0_k101/af_0_k101.mtx

.PHONY:all run test
all:
	/usr/local/cuda-11.8/bin/nvcc -O2 -arch=compute_80 -code=sm_80 ${EXE} -o ${OBJ}
run:all
	./${OBJ} ${input_Matrix} ${test_Matrix}
test:all
	nvprof ./${OBJ}

.PHONY:clean
clean:
	rm ${OBJ}