### CONFIGURATION BGEIN ###
# =======================
CUDA_INSTALL_PATH =/usr/local/cuda-6.0# set to your cuda installation path

SINGLE_GPU_ID	 = 0	# the GPU card id which you want to use (id is from 0 to #GPUs-1)
# =======================
### CONFIGURATION END ###


# modify the following options if you want to use multiple GPU cards or GPU together with CPU
# ==================
MULTICARD_ONLY_GPU	?= no	# yes if you want to use multiple GPU cards and not cooperate with CPU,if this entry is yes,HYBRID should be *no*
HYBRID			?= no	# yes if you want to use GPU together with CPU (only available if MULTICARD_ONLY_GPU is *no*)
MAXGPU			?= 1	# the maximum number of GPU you want to use
MAXCPU			?= 4	# how many CPU cores to use (zero if you want to use *GPU only*), only available if HYBRID is *yes*
SSE             	?= yes	# yes if your CPU support SSE, only available if HYBRID is *yes*
# ==================


CC = cc # cc, gcc, cl
LIBS = -lm # -lM
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
NVLIBS = -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_INSTALL_PATH)/lib -lcudart
NVFLAGS = -arch=sm_20


# for testing below,do not touch
# ==================
CUDA			?= yes	# use cuda or not
STREAM			?= yes	# use stream or not
DIVIDE_EQUALLY		?= yes	# yes if you want to divide the tasks among all GPU cards equally,only available if *MULTICARD_ONLY_GPU* is yes
DEBUG_RUN		?= no   
# ==================

PROG		= codeml

ifeq ($(strip $(CUDA)), yes)
    NVFLAGS	+= -DCUDA
    CFLAGS	+= -DCUDA
    LIBS	+= $(NVLIBS) -lstdc++
    OBJECTS	+= cuda-codeml.o
    PROG	= CuCodeML

ifeq ($(strip $(STREAM)), yes)
    NVFLAGS	+= -DSTREAM
    CFLAGS	+= -DSTREAM
endif

ifeq ($(strip $(MULTICARD_ONLY_GPU)), yes)
    NVFLAGS     += -DMULTICARD_ONLY_GPU -DHYBRID_OR_MULTICARD -DMAXGPU=${MAXGPU}
    CFLAGS      += -DMULTICARD_ONLY_GPU -DHYBRID_OR_MULTICARD -DMAXGPU=${MAXGPU}
    ifeq ($(strip $(DIVIDE_EQUALLY)), yes)
          NVFLAGS     += -DDIVIDE_EQUALLY
          CFLAGS      += -DDIVIDE_EQUALLY
    endif    
else           
ifeq ($(strip $(HYBRID)), yes)
    NVFLAGS	+= -DHYBRID -DHYBRID_OR_MULTICARD -DMAXGPU=${MAXGPU} -DMAXCPU=${MAXCPU} -Xcompiler -pthread -Xcompiler -fopenmp
    CFLAGS	+= -DHYBRID -DHYBRID_OR_MULTICARD -DMAXGPU=${MAXGPU} -DMAXCPU=${MAXCPU} -pthread -fopenmp
    LDFLAGS	+= -pthread -fopenmp
    OBJECTS	+= omp-codeml.o
else
    NVFLAGS	+= -DSINGLE_GPU_ID=${SINGLE_GPU_ID}
endif	# HYBRID
endif	# MULTICARD_ONLY_GPU

ifeq ($(strip $(SSE)), yes)
    CFLAGS	+= -DSSE
endif

endif # CUDA

ifeq ($(strip $(DEBUG_RUN)), yes)
    CFLAGS	+= -g -pg
    NVFLAGS	+= -g -G
else
    CFLAGS	+= -O3 -funroll-loops -fomit-frame-pointer
    NVFLAGS	+= -O3
endif

LDLIBS = $(LIBS)
LDFLAGS += $(CFLAGS)

OBJECTS += tools.o codeml.o
$(PROG) : $(OBJECTS)
	$(CC) $(LDFLAGS) $(LDLIBS) $(OBJECTS) -o $@

tools.o : tools.c paml.h
	$(CC) $(CFLAGS) -c tools.c
codeml.o : codeml.c treesub.c treespace.c paml.h codeml.h
	$(CC) $(CFLAGS) -c codeml.c

omp-codeml.o : omp-codeml.c
	$(CC) $(CFLAGS) -c omp-codeml.c 
cuda-codeml.o : cuda-codeml.cu codeml.h
	$(NVCC) $(NVFLAGS) -c cuda-codeml.cu

.PHONY : clean
clean :
	$(RM) CuCodeML $(OBJECTS)
