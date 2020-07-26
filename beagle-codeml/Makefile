### CONFIGURATION BGEIN ###
# =======================
CUDA_INSTALL_PATH =/usr/local/cuda-6.0# set to your cuda installation path
BEAGLE_INSTALL_PATH =/home/hxm/beagle_debug_sync# set this to your beagle installation path
BEAGLE_GPU_ID       ?=1# the GPU card you want to use (from 1 to #GPUs)
# =======================
### CONFIGURATION END ###

CC = cc # cc, gcc, cl
LIBS = -lm # -lM
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
NVLIBS = -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_INSTALL_PATH)/lib -lcudart
NVFLAGS = -arch=sm_20# modify this to match your GPU architecture
BEAGLELIBS = -L$(BEAGLE_INSTALL_PATH)/lib

# for testing below, do not touch
CUDA            ?= yes  # use cuda or not,should always be yes if you want to use GPU cards.
BEAGLE          ?= yes  # yes if you want to use beagle-codeml instead of cuda-codeml, only available if CUDA is *yes*
NOSCALE              ?= no	# do not scale in beagle-codeml (only available if BEAGLE is *yes*)
DEBUG_RUN            ?= no	# yes for debug purpose
# =================

PROG		= codeml

ifeq ($(strip $(CUDA)), yes)
    NVFLAGS	+= -DCUDA
    CFLAGS	+= -DCUDA 
    LIBS	+= $(NVLIBS) -lstdc++

 ifeq ($(strip $(BEAGLE)), yes)
   NVFLAGS     += -DBEAGLE -DBEAGLE_GPU_ID=${BEAGLE_GPU_ID}
   NVFLAGS     += -I$(BEAGLE_INSTALL_PATH)/include/libhmsbeagle-1
   CFLAGS      += -DBEAGLE
   CFLAGS      += -I$(BEAGLE_INSTALL_PATH)/include/libhmsbeagle-1
   LIBS        += $(BEAGLELIBS) -lhmsbeagle
   PROG         = BeagleCodeML
   OBJECTS     += beagle-codeml.o tools_beagle.o codeml_beagle.o
   
   ifeq ($(strip $(NOSCALE)), yes)
   NVFLAGS     += -DNOSCALE
   CFLAGS      += -DNOSCALE
   endif  
 endif	# BEAGLE
endif	# CUDA

ifeq ($(strip $(DEBUG_RUN)), yes)
    CFLAGS	+= -g -pg
    NVFLAGS	+= -g -G 
else
    CFLAGS	+= -O3 -funroll-loops -fomit-frame-pointer
    NVFLAGS	+= -O3
endif

LDLIBS = $(LIBS)
LDFLAGS += $(CFLAGS)

$(PROG) : $(OBJECTS)
	$(CC) $(LDFLAGS) $(LDLIBS) $(OBJECTS) -o $@

tools_beagle.o : tools_beagle.c paml_beagle.h
	$(CC) $(CFLAGS) -c tools_beagle.c
beagle-codeml.o : beagle-codeml.cu codeml_beagle.h
	$(NVCC) $(NVFLAGS) -c beagle-codeml.cu
codeml_beagle.o : codeml_beagle.c codeml_beagle.h treesub_beagle.c treespace_beagle.c paml_beagle.h 
	$(CC) $(CFLAGS) -c codeml_beagle.c

.PHONY : clean
clean :
	$(RM)  $(OBJECTS)
