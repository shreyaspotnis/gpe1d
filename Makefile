CUDA_INSTALL_PATH = $(SCINET_CUDA_INSTALL)
CUDA_FLAGS = -arch=sm_20 -O2
CUDA_INC = -I$(SCINET_CUDA_INC)
CUDA_LIB = -L$(SCINET_CUDA_LIB)
LIBS = -lcufft 

all:
	nvcc $(CUDA_FLAGS) -o gpec_cuda gpec_cuda.cu pca_utils.c $(CUDA_INC) $(CUDA_LIB) $(LIBS)

