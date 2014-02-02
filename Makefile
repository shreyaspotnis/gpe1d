CUDA_INSTALL_PATH = $(SCINET_CUDA_INSTALL)
CUDA_FLAGS = -arch=sm_20 -O2
CUDA_INC = -I$(SCINET_CUDA_INC)
CUDA_LIB = -L$(SCINET_CUDA_LIB)
LIBS = -lcufft 

all: bin/gpe1d_cuda_float bin/gpe1d_cuda_double

bin/gpe1d_cuda_float: src_cuda/gpe1d.cu src_cuda/pca_utils.c
	nvcc $(CUDA_FLAGS) -o bin/gpe1d_cuda_float src_cuda/gpe1d.cu src_cuda/pca_utils.c $(CUDA_INC) $(CUDA_LIB) $(LIBS)

bin/gpe1d_cuda_double: src_cuda/gpe1d.cu src_cuda/pca_utils.c
	nvcc $(CUDA_FLAGS) -D_USE_DOUBLE_PRECISION -o bin/gpe1d_cuda_double src_cuda/gpe1d.cu src_cuda/pca_utils.c $(CUDA_INC) $(CUDA_LIB) $(LIBS)

clean:
	rm bin/gpe1d_cuda_float bin/gpe1d_cuda_double

