EXE = spmm.cu
OBJ = $(patsubst %.cu,%, $(EXE))
#log_PATH = ${log_path}
log_PATH = /data/seery/src/log/cuda.log

input_Matrix = ${input_matrix}
#input_Matrix = "none"
test_Matrix =${test_matrix}
#test_Matrix = /data/suitesparse_collection/bcsstk18/bcsstk18.mtx

NVCC_FLAGS = -O2 -std=c++17 --use_fast_math -gencode arch=compute_80,code=sm_80

.PHONY:all run test
all:
	/usr/local/cuda-11.8/bin/nvcc ${NVCC_FLAGS} ${EXE} -o ${OBJ}
run:all
	./${OBJ} ${input_Matrix} ${test_Matrix} > ${log_PATH}
test:all
	nvprof ./${OBJ}

.PHONY:clean
clean:
	rm ${OBJ}