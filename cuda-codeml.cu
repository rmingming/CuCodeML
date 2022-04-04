/*
  cuda-codeml.cu
  provides the CUDA version of calculating transition matrix and conditional likelihood.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#include "paml.h"
#include "codeml.h"

#define shTILE 8                      /* tile size */


#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

#define cutilCheckMsg(msg)           __cutilGetLastError (msg, __FILE__, __LINE__)
inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

int global_ir;

double PMatT[NNODE];                            /* branch length */
int PMatiUVR[NNODE];                            /* which set of U,V,R to use(there are 3 sets in total) */
double __attribute((aligned(0x10))) extendUVR[3 * UVR_SIZE];                  /* extend U,V to 64*64 and R to 64 */      

int bfs_que[NS];                                     
int tree_layer[NS];
int layer_cnt;

int father[NNODE];                               /* father node of the current node */
int first_son[NNODE];                            /* first son of the current node */
int schedule_que[NNODE * 2];                        /* the order of calculating contitional probability after schedule so as to maxmimum parallism */
int schedule_scale[NNODE * 2];                     /* mark whether the current node needs scaling */
int nodeScaleK[NNODE];

#ifdef STREAM
cudaStream_t stream[MAXCARD][NS];                             
#endif

#ifdef MULTICARD_ONLY_GPU                          /* MULTICARD_ONLY_GPU: only GPUs do the calculation if there are multiple cards */
int dev_count;
pthread_t threads[MAXCARD];
int thread_ig;
int thread_pos0[NGENE][MAXCARD];                   
int thread_pos1[NGENE][MAXCARD];
#else
#ifdef HYBRID                                      /* HYBRID: both GPU and CPU do the calculation */
int dev_count;
pthread_t threads[MAXCARD + 1];
int thread_ig;
int thread_pos0[NGENE][MAXCARD + 1];
int thread_pos1[NGENE][MAXCARD + 1];

double __attribute((aligned(0x10))) PMatAll[NNODE][64 * 64];
double * nodes_conP[NNODE];
#endif
#endif

__constant__ int DEV_tree_root;
__constant__ int DEV_com_ncode;
__constant__ int DEV_com_ls;
__constant__ int DEV_com_npatt;
__constant__ int DEV_com_npatt_extend;                          /* DEV_com_npatt_extend: the extended length of com.npatt,which is a multiple of 32 */
__constant__ int DEV_com_NnodeScale;
__constant__ char DEV_nChara[256], DEV_CharaMap[256][64];
__constant__ double DEV_com_pi[NCODE];

char *dev_com_z[MAXCARD];		   // com.ns * com.ls  
double *dev_com_nodeScaleF[MAXCARD];	// com.NnodeScale * com.npatt
double *dev_com_conP[MAXCARD];		
double *dev_com_fhK[MAXCARD];		// com.npatt
double *dev_PMat[MAXCARD];		
double *dev_UVR[MAXCARD];
double *dev_exptRootAll[MAXCARD];
double *dev_PMatT[MAXCARD];
int *dev_PMatiUVR[MAXCARD];


__device__
inline void saxpy(double a, const double *b, double *c, int n)
{
/*
  Calculate c[] += a * b[], later used in calculating transition matrix.
*/
#pragma unroll
    for (int i = 0; i < n; i++) {
        c[i] += a * b[i];
    }
}

__global__
void kernelConditionalPNodeTipClean(int pos0, int pos1, int inode, int ison, const double *PMat, const char *com_z, double *conP)
{
/*
  Kernel of calculating conditional probability of the node whose current son(ison) is a tip node and the sequence is clean.
*/
    const int h = blockIdx.x + pos0;
    const int j = threadIdx.x;
	const int n = 64;
    const int m = DEV_com_npatt_extend;
    const double *PMatIson = PMat + ison * 64 * 64;

    conP[inode * n * m + h * n + j] *= PMatIson[com_z[ison * DEV_com_ls + h] * 64 + j];
}

__global__
void kernelConditionalPNodeTipCleanFirst(int pos0, int pos1, int inode, int ison, const double *PMat, const char *com_z, double *conP)
{
/*
  Kernel of calculating conditional probability of the node whose current son(ison) is a tip node and the first son of the father node with the sequence clean.
*/
    const int h = blockIdx.x + pos0;
    const int j = threadIdx.x;
	const int n = 64;
    const int m = DEV_com_npatt_extend;
    const double *PMatIson = PMat + ison * 64 * 64;

    conP[inode * n * m + h * n + j] = PMatIson[com_z[ison * DEV_com_ls + h] * 64 + j];
}

void callKernelConditionalPNodeTipClean(int dev_id, int pos0, int pos1, int inode, int ison)
{
/*
  Call one of the above two kernels according to whether the son node(ison) is the first son or not.
  If STREAM is defined, use the stream to provide more parallelism.
*/
    int threadsPerBlock = com.ncode;
    int blocksPerGrid = pos1 - pos0;
    if (first_son[ison]) {
#ifdef STREAM
        kernelConditionalPNodeTipCleanFirst<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#else
        kernelConditionalPNodeTipCleanFirst<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#endif
    } else {
#ifdef STREAM
        kernelConditionalPNodeTipClean<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#else
        kernelConditionalPNodeTipClean<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#endif
    }
    cutilCheckMsg("cudaKernel kernelConditionalPNodeTipClean failed");
}

__global__
void kernelConditionalPNodeTipUnclean(int pos0, int pos1, int inode, int ison, const double *PMat, const char *com_z, double *conP)
{
/*
  Kernel of calculating the conditional probability of the node whose current son(ison) is the tip node and the sequence is unclean.
*/
    const int h = blockIdx.x + pos0;
    const int j = threadIdx.x;
	const int n = 64;
    const int m = DEV_com_npatt_extend;
    const double *PMatIson = PMat + ison * 64 * 64;

    double t = 0;
    int k;
    for (k = 0; k < DEV_nChara[com_z[ison * DEV_com_ls + h]]; k++) {
        t += PMatIson[DEV_CharaMap[com_z[ison * DEV_com_ls + h]][k] * 64 + j];
    }
    conP[inode * n * m + h * n + j] *= t;
}

__global__
void kernelConditionalPNodeTipUncleanFirst(int pos0, int pos1, int inode, int ison, const double *PMat, const char *com_z, double *conP)
{
/*
  Kernel of calculating the conditional probability of the node whose current son(ison) is the tip node and the first son with the sequence unclean.
*/
    const int h = blockIdx.x + pos0;
    const int j = threadIdx.x;
	const int n = 64;
    const int m = DEV_com_npatt_extend;
    const double *PMatIson = PMat + ison * 64 * 64;

    double t = 0;
    int k;
    for (k = 0; k < DEV_nChara[com_z[ison * DEV_com_ls + h]]; k++) {
        t += PMatIson[DEV_CharaMap[com_z[ison * DEV_com_ls + h]][k] * 64 + j];
    }
    conP[inode * n * m + h * n + j] = t;
}

void callKernelConditionalPNodeTipUnclean(int dev_id, int pos0, int pos1, int inode, int ison)
{
    int threadsPerBlock = com.ncode;
    int blocksPerGrid = pos1 - pos0;
    if (first_son[ison]) {
#ifdef STREAM
        kernelConditionalPNodeTipUncleanFirst<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#else
        kernelConditionalPNodeTipUncleanFirst<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#endif
    } else {
#ifdef STREAM
        kernelConditionalPNodeTipUnclean<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#else
        kernelConditionalPNodeTipUnclean<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_z[dev_id], dev_com_conP[dev_id]);
#endif
    }
    cutilCheckMsg("cudaKernel kernelConditionalPNodeTipUnclean failed");
}


#if __CUDA_ARCH__ >= 350                               /* if arch>=35, using __ldg() to load data from global memory */
__global__
void kernelConditionalPNodeInternal(int pos0, int pos1, int inode, int ison, const double *PMat, double *conP)
{
/*
  Kernel of calculating the conditional probability of the node whose current son(ison) is an internal node.
*/
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
	const int k = DEV_com_npatt_extend;
	const int ttx = 2*tx;
	const int tty = 2*ty;
	double * conPIson = conP + ison * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + tx;
	PMat += ison * 64 * 64 +  ty * 64 + blockIdx.x * 32 + ttx ;
    __shared__ double conPs[32][shTILE];
	__shared__ double PMats[shTILE][32];
	double s[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double buf[shTILE] = {0, 0, 0, 0, 0, 0, 0, 0};

	int itr,i;
	for(itr=0; itr<64/shTILE; itr++){
        buf[0] = __ldg(&conPIson[0]);
		buf[1] = __ldg(&conPIson[64]);
		buf[2] = __ldg(&conPIson[16*64]);
		buf[3] = __ldg(&conPIson[17*64]);

		buf[4] = __ldg(&PMat[0]);
		buf[5] = __ldg(&PMat[1]);
		buf[6] = __ldg(&PMat[16]);
		buf[7] = __ldg(&PMat[17]);

        conPs[tty][tx] = buf[0];
		conPs[tty+1][tx] = buf[1];
		conPs[tty+16][tx] = buf[2];
		conPs[tty+17][tx] = buf[3];

		PMats[ty][ttx] = buf[4];
		PMats[ty][ttx+1] = buf[5];
		PMats[ty][ttx+16] = buf[6];
		PMats[ty][ttx+17] = buf[7];
        __syncthreads();

	    for(i=0;i<shTILE;i++){
	       s[0] += conPs[tty][i] * PMats[i][ttx]; 
	       s[1] += conPs[tty+1][i] * PMats[i][ttx+1];
	       s[2] += conPs[tty][i] * PMats[i][ttx+1];
	       s[3] += conPs[tty+1][i] * PMats[i][ttx];

	       s[4] += conPs[tty][i] * PMats[i][ttx+16];
	       s[5] += conPs[tty+1][i] * PMats[i][ttx+17];
	       s[6] += conPs[tty][i] * PMats[i][ttx+17];
	       s[7] += conPs[tty+1][i] * PMats[i][ttx+16];
	   
	       s[8] += conPs[tty+16][i] * PMats[i][ttx];
	       s[9] += conPs[tty+17][i] * PMats[i][ttx+1];
	       s[10] += conPs[tty+16][i] * PMats[i][ttx+1];
	       s[11] += conPs[tty+17][i] * PMats[i][ttx];

	       s[12] += conPs[tty+16][i] * PMats[i][ttx+16];
	       s[13] += conPs[tty+17][i] * PMats[i][ttx+17];
	       s[14] += conPs[tty+16][i] * PMats[i][ttx+17];
	       s[15] += conPs[tty+17][i] * PMats[i][ttx+16];   
	   }
       __syncthreads();
       conPIson += shTILE;
       PMat += shTILE * 64;
    }
	conP += inode * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + blockIdx.x * 32 + ttx;
	conP[0] *= s[0];
	conP[1] *= s[2];
	conP[16] *= s[4];
    conP[17] *= s[6];
	 
	conP[64] *= s[3];
	conP[64+1] *= s[1];
	conP[64+16] *= s[7];
	conP[64+17] *= s[5];

	conP[16*64] *= s[8];
	conP[16*64+1] *= s[10];
	conP[16*64+16] *= s[12];
	conP[16*64+17] *= s[14];

	conP[17*64] *= s[11];
	conP[17*64+1] *= s[9];
	conP[17*64+16] *= s[15];
	conP[17*64+17] *= s[13];
}

__global__
void kernelConditionalPNodeInternalFirst(int pos0, int pos1, int inode, int ison, const double *PMat, double *conP)
{
/*
  Kernel of calculating the conditional probability whose current son node(ison) is an internal node and the first son.
*/
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
	const int k = DEV_com_npatt_extend;
	const int ttx = 2 * tx;
	const int tty = 2 * ty;
	double * conPIson = conP + ison * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + tx;
	PMat += ison * 64 * 64 +  ty * 64 + blockIdx.x * 32 + ttx ;
    __shared__ double conPs[32][shTILE];
	__shared__ double PMats[shTILE][32];
	double s[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double buf[shTILE] = {0, 0, 0, 0, 0, 0, 0, 0};

	int itr,i;
	for(itr=0; itr<64/shTILE; itr++){
		buf[0] = __ldg(&conPIson[0]);
		buf[1] = __ldg(&conPIson[64]);
		buf[2] = __ldg(&conPIson[16*64]);
		buf[3] = __ldg(&conPIson[17*64]);

		buf[4] = __ldg(&PMat[0]);
		buf[5] = __ldg(&PMat[1]);
		buf[6] = __ldg(&PMat[16]);
		buf[7] = __ldg(&PMat[17]);

        conPs[tty][tx] = buf[0];
		conPs[tty+1][tx] = buf[1];
		conPs[tty+16][tx] = buf[2];
		conPs[tty+17][tx] = buf[3];

		PMats[ty][ttx] = buf[4];
		PMats[ty][ttx+1] = buf[5];
		PMats[ty][ttx+16] = buf[6];
		PMats[ty][ttx+17] = buf[7];
        __syncthreads();

	    for(i=0;i<shTILE;i++){
	      s[0] += conPs[tty][i]*PMats[i][ttx]; 
	      s[1] += conPs[tty+1][i]*PMats[i][ttx+1];
	      s[2] += conPs[tty][i]*PMats[i][ttx+1];
	      s[3] += conPs[tty+1][i]*PMats[i][ttx];

	      s[4] += conPs[tty][i]*PMats[i][ttx+16];
	      s[5] += conPs[tty+1][i]*PMats[i][ttx+17];
	      s[6] += conPs[tty][i]*PMats[i][ttx+17];
	      s[7] += conPs[tty+1][i]*PMats[i][ttx+16];
	   
	      s[8] += conPs[tty+16][i]*PMats[i][ttx];
	      s[9] += conPs[tty+17][i]*PMats[i][ttx+1];
	      s[10] += conPs[tty+16][i]*PMats[i][ttx+1];
	      s[11] += conPs[tty+17][i]*PMats[i][ttx];

	      s[12] += conPs[tty+16][i]*PMats[i][ttx+16];
	      s[13] += conPs[tty+17][i]*PMats[i][ttx+17];
	      s[14] += conPs[tty+16][i]*PMats[i][ttx+17];
	      s[15] += conPs[tty+17][i]*PMats[i][ttx+16];
	  }
      __syncthreads();
      conPIson += shTILE;
      PMat += shTILE * 64;
  }
	
  conP += inode * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + blockIdx.x * 32 + ttx;
  conP[0] = s[0];
  conP[1] = s[2];
  conP[16] = s[4];
  conP[17] = s[6];

  conP[64] = s[3];
  conP[64+1] = s[1];
  conP[64+16] = s[7];
  conP[64+17] = s[5];

  conP[16*64] = s[8];
  conP[16*64+1] = s[10];
  conP[16*64+16] = s[12];
  conP[16*64+17] = s[14];

  conP[17*64] = s[11];
  conP[17*64+1] = s[9];
  conP[17*64+16] = s[15];
  conP[17*64+17] = s[13];
}
#else             
__global__
void kernelConditionalPNodeInternal(int pos0, int pos1, int inode, int ison, const double *PMat, double *conP)
{
/*
  Kernel of calculating the conditional probability of the node whose current son(ison) is an internal node.
*/
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
	const int k = DEV_com_npatt_extend;
	const int ttx = 2*tx;
	const int tty = 2*ty;
	double * conPIson = conP + ison * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + tx;
	PMat += ison * 64 * 64 +  ty * 64 + blockIdx.x * 32 + ttx ;
    __shared__ double conPs[32][shTILE];
	__shared__ double PMats[shTILE][32];
	double s[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	int itr,i;
	for(itr=0; itr<64/shTILE; itr++){
	    conPs[tty][tx] = conPIson[0];
		conPs[tty+1][tx] = conPIson[64];
		conPs[tty+16][tx] = conPIson[16*64];
		conPs[tty+17][tx] = conPIson[17*64];

		PMats[ty][ttx] = PMat[0];
		PMats[ty][ttx+1] = PMat[1];
		PMats[ty][ttx+16] = PMat[16];
		PMats[ty][ttx+17] = PMat[17];
        __syncthreads();

	    for(i=0;i<shTILE;i++){
	       s[0] += conPs[tty][i] * PMats[i][ttx]; 
	       s[1] += conPs[tty+1][i] * PMats[i][ttx+1];
	       s[2] += conPs[tty][i] * PMats[i][ttx+1];
	       s[3] += conPs[tty+1][i] * PMats[i][ttx];

	       s[4] += conPs[tty][i] * PMats[i][ttx+16];
	       s[5] += conPs[tty+1][i] * PMats[i][ttx+17];
	       s[6] += conPs[tty][i] * PMats[i][ttx+17];
	       s[7] += conPs[tty+1][i] * PMats[i][ttx+16];
	   
	       s[8] += conPs[tty+16][i] * PMats[i][ttx];
	       s[9] += conPs[tty+17][i] * PMats[i][ttx+1];
	       s[10] += conPs[tty+16][i] * PMats[i][ttx+1];
	       s[11] += conPs[tty+17][i] * PMats[i][ttx];

	       s[12] += conPs[tty+16][i] * PMats[i][ttx+16];
	       s[13] += conPs[tty+17][i] * PMats[i][ttx+17];
	       s[14] += conPs[tty+16][i] * PMats[i][ttx+17];
	       s[15] += conPs[tty+17][i] * PMats[i][ttx+16];   
	   }
       __syncthreads();
       conPIson += shTILE;
       PMat += shTILE * 64;
    }
	conP += inode * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + blockIdx.x * 32 + ttx;
	conP[0] *= s[0];
	conP[1] *= s[2];
	conP[16] *= s[4];
    conP[17] *= s[6];
	 
	conP[64] *= s[3];
	conP[64+1] *= s[1];
	conP[64+16] *= s[7];
	conP[64+17] *= s[5];

	conP[16*64] *= s[8];
	conP[16*64+1] *= s[10];
	conP[16*64+16] *= s[12];
	conP[16*64+17] *= s[14];

	conP[17*64] *= s[11];
	conP[17*64+1] *= s[9];
	conP[17*64+16] *= s[15];
	conP[17*64+17] *= s[13];
}

__global__
void kernelConditionalPNodeInternalFirst(int pos0, int pos1, int inode, int ison, const double *PMat, double *conP)
{
/*
  Kernel of calculating the conditional probability whose current son node(ison) is an internal node and the first son.
*/
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
	const int k = DEV_com_npatt_extend;
	const int ttx = 2 * tx;
	const int tty = 2 * ty;
	double * conPIson = conP + ison * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + tx;
	PMat += ison * 64 * 64 +  ty * 64 + blockIdx.x * 32 + ttx ;
    __shared__ double conPs[32][shTILE];
	__shared__ double PMats[shTILE][32];
	double s[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	int itr,i;
	for(itr=0; itr<64/shTILE; itr++){
		conPs[tty][tx] = conPIson[0];
		conPs[tty+1][tx] = conPIson[64];
		conPs[tty+16][tx] = conPIson[16*64];
		conPs[tty+17][tx] = conPIson[17*64];

		PMats[ty][ttx] = PMat[0];
		PMats[ty][ttx+1] = PMat[1];
		PMats[ty][ttx+16] = PMat[16];
		PMats[ty][ttx+17] = PMat[17];
        __syncthreads();

	    for(i=0;i<shTILE;i++){
	      s[0] += conPs[tty][i]*PMats[i][ttx]; 
	      s[1] += conPs[tty+1][i]*PMats[i][ttx+1];
	      s[2] += conPs[tty][i]*PMats[i][ttx+1];
	      s[3] += conPs[tty+1][i]*PMats[i][ttx];

	      s[4] += conPs[tty][i]*PMats[i][ttx+16];
	      s[5] += conPs[tty+1][i]*PMats[i][ttx+17];
	      s[6] += conPs[tty][i]*PMats[i][ttx+17];
	      s[7] += conPs[tty+1][i]*PMats[i][ttx+16];
	   
	      s[8] += conPs[tty+16][i]*PMats[i][ttx];
	      s[9] += conPs[tty+17][i]*PMats[i][ttx+1];
	      s[10] += conPs[tty+16][i]*PMats[i][ttx+1];
	      s[11] += conPs[tty+17][i]*PMats[i][ttx];

	      s[12] += conPs[tty+16][i]*PMats[i][ttx+16];
	      s[13] += conPs[tty+17][i]*PMats[i][ttx+17];
	      s[14] += conPs[tty+16][i]*PMats[i][ttx+17];
	      s[15] += conPs[tty+17][i]*PMats[i][ttx+16];
	  }
      __syncthreads();
      conPIson += shTILE;
      PMat += shTILE * 64;
  }
	
  conP += inode * k * 64 + (pos0 + blockIdx.y * 32 + tty) * 64 + blockIdx.x * 32 + ttx;
  conP[0] = s[0];
  conP[1] = s[2];
  conP[16] = s[4];
  conP[17] = s[6];

  conP[64] = s[3];
  conP[64+1] = s[1];
  conP[64+16] = s[7];
  conP[64+17] = s[5];

  conP[16*64] = s[8];
  conP[16*64+1] = s[10];
  conP[16*64+16] = s[12];
  conP[16*64+17] = s[14];

  conP[17*64] = s[11];
  conP[17*64+1] = s[9];
  conP[17*64+16] = s[15];
  conP[17*64+17] = s[13];
}
#endif              


void callKernelConditionalPNodeInternal(int dev_id, int pos0, int pos1, int inode, int ison)
{
	int bly ;
	if((pos1-pos0)%32==0){
	   bly = (pos1-pos0)/32;
	}
	else{
	   bly = (pos1-pos0)/32+1;
	}
    dim3 blocksPerGrid(2,bly);
	dim3 threadsPerBlock(8,8);
    if (first_son[ison]) {
#ifdef STREAM
        kernelConditionalPNodeInternalFirst<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_conP[dev_id]);
#else
        kernelConditionalPNodeInternalFirst<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_conP[dev_id]);
#endif
    }else{ 	
#ifdef STREAM
        kernelConditionalPNodeInternal<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_conP[dev_id]);
#else
        kernelConditionalPNodeInternal<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, inode, ison, dev_PMat[dev_id], dev_com_conP[dev_id]);
#endif
    }
    cutilCheckMsg("cudaKernel kernelConditionalPNodeInternal failed");
}


__global__
void kernelNodeScale(int inode, int pos0, int pos1, int k, double *conP, double *nodeScaleF)
{
/*
  Kernel of scaling the conditional probability to avoid underflow.
*/
    const int h = pos0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= pos1) return ;

	const int n = 64;
    const int m = DEV_com_npatt;
	const int me = DEV_com_npatt_extend;

    int j;
    double t = 0;
    for (j = 0; j < DEV_com_ncode; j++) {
        t = MAX(t, conP[inode * n * me + h * n + j]);
    }

    if (t < 1e-300) {
        for (j = 0; j < DEV_com_ncode; j++)
            conP[inode * n * me + h * n + j] = 1;
        nodeScaleF[k * m + h] = -800;  /* this is problematic? */
    } else {
        double tt = 1 / t;
        for (j = 0; j < DEV_com_ncode; j++)
            conP[inode * n * me + h * n + j] *= tt;
        nodeScaleF[k * m + h] = log(t);
    }
}

void callKernelNodeScale(int inode, int dev_id, int pos0, int pos1, int k)
{
    int cnt = pos1 - pos0;
    int threadsPerBlock = 128;
    int blocksPerGrid = (cnt + threadsPerBlock - 1) / threadsPerBlock;
#ifdef STREAM
    kernelNodeScale<<<blocksPerGrid, threadsPerBlock, 0, stream[dev_id][inode - com.ns]>>>(inode, pos0, pos1, k, dev_com_conP[dev_id], dev_com_nodeScaleF[dev_id]);
#else
    kernelNodeScale<<<blocksPerGrid, threadsPerBlock>>>(inode, pos0, pos1, k, dev_com_conP[dev_id], dev_com_nodeScaleF[dev_id]);
#endif
    cutilCheckMsg("cudaKernel kernelNodeScale failed");
}



__global__
void kernelCalcFhK(int pos0, int pos1, const double *conP, const double *nodeScaleF, double *fhK)
{
/*
  Kernel of intergrating the conditional probability of the root.
*/
    const int h = pos0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= pos1) return ;

	const int n = 64;
    const int m = DEV_com_npatt;
	const int me = DEV_com_npatt_extend;
    const double *rootConP = conP + DEV_tree_root * n * me;

    double fh = 0;
    int i;
    for (i = 0; i < DEV_com_ncode; i++) {
        fh += DEV_com_pi[i] * rootConP[h * n + i];
    }
    if (fh <= 0) {
        fh = 1e-300;
    }
    if (!DEV_com_NnodeScale) {
        fhK[h] = fh;
    } else {
        fhK[h] = log(fh);
        int k;
        for (k = 0; k < DEV_com_NnodeScale; k++) {
            fhK[h] += nodeScaleF[k * m + h];
        }
    }
}

void callKernelCalcFhK(int dev_id, int pos0, int pos1)
{
    int cnt = pos1 - pos0;
    int threadsPerBlock = 128;
    int blocksPerGrid = (cnt + threadsPerBlock - 1) / threadsPerBlock;
    kernelCalcFhK<<<blocksPerGrid, threadsPerBlock>>>(pos0, pos1, dev_com_conP[dev_id], dev_com_nodeScaleF[dev_id], dev_com_fhK[dev_id]);
    cutilCheckMsg("cudaKernel kernelCalcFhK failed");
}


__global__
void kernelPMatExptRoot(const double *PMatT, const int *PMatiUVR, const double *UVR, double *exptRootAll)
{
/*
  Kernel of calculating exptRootAll[]=exp(t*UVR[]);
*/
    int inode = blockIdx.x;
    if (inode == DEV_tree_root) return ;

    int tx = threadIdx.x;
    int idx = inode * 64 + tx;
    double t = PMatT[inode];
    int ridx = PMatiUVR[inode] * UVR_SIZE + 64 * 64 * 2 + tx;
    exptRootAll[idx] = exp(t * UVR[ridx]);
}


__global__
void kernelPMatUVRoot(const int *PMatiUVR, const double *UVR, const double *exptRootAll, double *PMat)
{
/*
  Kernel of calculating transition matrix.
*/
    const int inode = blockIdx.y;
    if (inode == DEV_tree_root) return ;

    const int n = 64, m = 16;
    const int id = threadIdx.x;
    const int ibx = blockIdx.x * m;

    const double *U = UVR + PMatiUVR[inode] * UVR_SIZE;
    const double *V = U + n * n;
    double *P = PMat + inode * n * n;

    U += ibx * n;
    V += id;
    P += id * n + ibx;

    double exptRoot = 0;
    if (id < DEV_com_ncode) exptRoot = exptRootAll[inode * 64 + id];

    __shared__ double bs[n][m + 1];
#pragma unroll
    for (int i = 0; i < m; i++) {
        bs[id][i] = U[i * n + id] * exptRoot;
    }
    __syncthreads();

    double c[m] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double v0, v1 = V[0];
#pragma unroll
    for (int i = 0; i < n; i++) {
        v0 = v1;
        V += n;
        v1 = V[0];
        saxpy(v0, bs[i], c, 16);
    }

#pragma unroll
    for (int i = 0; i < m; i++) {
        if (c[i] < 0) c[i] = 0;
        P[i] = c[i];
    }
}

void callKernelPMatUVRootAll(int dev_id)
{
    cutilSafeCall(cudaMemcpy(dev_UVR[dev_id], extendUVR, 3 * UVR_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpy dev_UVR[dev_id] failed");
    cutilSafeCall(cudaMemcpy(dev_PMatT[dev_id], PMatT, tree.nnode * sizeof(double), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpy dev_PMatT failed");
    cutilSafeCall(cudaMemcpy(dev_PMatiUVR[dev_id], PMatiUVR, tree.nnode * sizeof(int), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpy dev_PMatiUVR failed");

    kernelPMatExptRoot<<<tree.nnode, com.ncode>>>(dev_PMatT[dev_id], dev_PMatiUVR[dev_id], dev_UVR[dev_id], dev_exptRootAll[dev_id]);
    cutilCheckMsg("cudaKernel kernelPMatExptRoot failed");

    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid(4, tree.nnode);
    kernelPMatUVRoot<<<blocksPerGrid, threadsPerBlock>>>(dev_PMatiUVR[dev_id], dev_UVR[dev_id], dev_exptRootAll[dev_id], dev_PMat[dev_id]);
    cutilCheckMsg("kernelPMatUVRoot failed!");
}

void deviceInit(int dev_id)
{
/*
  Initialize the device specified by dev_id.
*/
    int i;

    cutilSafeCall(cudaSetDevice(dev_id));
    cutilCheckMsg("cudaSetDevice failed");

    cutilSafeCall(cudaMemcpyToSymbol(DEV_tree_root, &tree.root, sizeof(int), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_tree_root failed");

    cutilSafeCall(cudaMemcpyToSymbol(DEV_com_ncode, &com.ncode, sizeof(int), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_com_ncode failed");
    cutilSafeCall(cudaMemcpyToSymbol(DEV_com_ls, &com.ls, sizeof(int), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_com_ls failed");
    cutilSafeCall(cudaMemcpyToSymbol(DEV_com_npatt, &com.npatt, sizeof(int), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_com_npatt failed");

	int npatt_extend=com.npatt;
	if(com.npatt%32!=0){
	   npatt_extend = (com.npatt/32+1)*32;
	}

	cutilSafeCall(cudaMemcpyToSymbol(DEV_com_npatt_extend, &npatt_extend, sizeof(int), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_com_npatt failed");
    cutilSafeCall(cudaMemcpyToSymbol(DEV_com_NnodeScale, &com.NnodeScale, sizeof(int), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_com_NnodeScale failed");

    cutilSafeCall(cudaMemcpyToSymbol(DEV_nChara, nChara, sizeof(nChara), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_nChara failed");
    cutilSafeCall(cudaMemcpyToSymbol(DEV_CharaMap, CharaMap, sizeof(CharaMap), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_CharaMap failed");

    cutilSafeCall(cudaMemcpyToSymbol(DEV_com_pi, &com.pi, sizeof(com.pi), 0, cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpyToSymbol DEV_com_pi failed");

    cutilSafeCall(cudaMalloc((void **)&dev_com_z[dev_id], com.ns * com.ls * sizeof(char)));
    cutilCheckMsg("cudaMalloc dev_com_z failed");
    for (i = 0; i < com.ns; i++) {
        cutilSafeCall(cudaMemcpy(dev_com_z[dev_id] + com.ls * i, com.z[i], com.ls * sizeof(char), cudaMemcpyHostToDevice));
        cutilCheckMsg("cudaMemcpy dev_com_z failed");
    }

    cutilSafeCall(cudaMalloc((void **)&dev_com_nodeScaleF[dev_id], com.NnodeScale * com.npatt * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_com_nodeScaleF failed");

    cutilSafeCall(cudaMalloc((void **)&dev_com_conP[dev_id], tree.nnode * 64 * npatt_extend * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_com_conP failed");

	cutilSafeCall(cudaMemset(dev_com_conP[dev_id], 0, tree.nnode * 64 * npatt_extend * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_com_conP failed");


    cutilSafeCall(cudaMalloc((void **)&dev_com_fhK[dev_id], com.npatt * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_fhK failed");

    cutilSafeCall(cudaMalloc((void **)&dev_PMat[dev_id], tree.nnode * 64 * 64 * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_PMat failed");
    cutilSafeCall(cudaMalloc((void **)&dev_UVR[dev_id], 3 * (64 * 64 * 2 + 64) * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_UVR failed");
    cutilSafeCall(cudaMalloc((void **)&dev_exptRootAll[dev_id], tree.nnode * 64 * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_exptRootAll failed");
    cutilSafeCall(cudaMalloc((void **)&dev_PMatT[dev_id], tree.nnode * sizeof(double)));
    cutilCheckMsg("cudaMalloc dev_PMatT failed");
    cutilSafeCall(cudaMalloc((void **)&dev_PMatiUVR[dev_id], tree.nnode * sizeof(int)));
    cutilCheckMsg("cudaMalloc dev_PMatiUVR failed");

#ifdef STREAM
    for (i = 0; i < com.ns; i++) {
        cutilSafeCall(cudaStreamCreate(&stream[dev_id][i]));
        cutilCheckMsg("cudaStreamCreate failed");
    }
#endif
}

int treeSonsCompare(const void *a, const void *b)
{
/*
  Compare the number of sons of the two nodes specified by a and b , return 1 if the a has more sons than b.
*/
    const int x = *(const int *)a;
    const int y = *(const int *)b;
    if (schedule_scale[x]) return 1;
    if (schedule_scale[y]) return -1;
    return nodes[x].nson - nodes[y].nson;
}

void prepareTree()
{
/*
  Schedule the order the nodes are traversed and mark the index where the node needs scaling.
*/
    int head = 0, tail = 0;

    bfs_que[tail++] = tree.root;
    tree_layer[0] = 0;
    tree_layer[1] = 1;
    layer_cnt = 2;
    while (head < tail) {
        if (head == tree_layer[layer_cnt - 1]) tree_layer[layer_cnt++] = tail;

        int node = bfs_que[head++];
        int i;
        int left = tail;
        for (i = 0; i < nodes[node].nson; i++) {
            int son = nodes[node].sons[i];
            father[son] = node;
            bfs_que[tail++] = son;
        }
        int right = tail;
        if (right - left) {
            qsort(bfs_que + left, right - left, sizeof(int), treeSonsCompare);
        }
    }
    assert(tail == tree.nnode);

    memset(first_son, 0, sizeof first_son);
    memset(schedule_scale, 0, sizeof schedule_scale);
    tail = 0;
    int scale_cnt = 0;
    int layer;
    for (layer = layer_cnt - 1; layer > 0; layer--) {
        int i;
        int maxRound = 0;
        for (i = tree_layer[layer - 1]; i < tree_layer[layer]; i++) {
            int t = nodes[bfs_que[i]].nson + 1;
            if (t > maxRound) maxRound = t;
        }

        int round;
        for (round = 0; round < maxRound; round++) {
            int left = tail;
            for (i = tree_layer[layer - 1]; i < tree_layer[layer]; i++) {
                int inode = bfs_que[i];

                if (round < nodes[inode].nson) {
                    int ison = nodes[inode].sons[round];
                    schedule_que[tail++] = ison;
                    if (round == 0) first_son[ison] = 1;
                } else if (round == nodes[inode].nson && nodes[inode].nson) {
                    if(com.NnodeScale && com.nodeScale[inode]) {
                        schedule_scale[tail] = 1;
                        schedule_que[tail++] = inode;
                        nodeScaleK[inode] = scale_cnt++;
                    }
                }
            }
            int right = tail;
            if (right - left) {
                qsort(schedule_que + left, right - left, sizeof(int), treeSonsCompare);
            }
        }
    }
    assert(scale_cnt == com.NnodeScale);
    assert(tail == tree.nnode + com.NnodeScale - 1);
}

#ifdef HYBRID_OR_MULTICARD                                       /* HYBRID_OR_MULTICARD is defined if either MULTICARD_ONLY_GPU or HYBRID is defined */
long long posTest(int ig, double *x, int count)
{
    struct timeval tpBegin;
    gettimeofday(&tpBegin, NULL);

    while (count--) {
        CUDA_ScheduleConditionalPNode(0, ig, x);
    }

    struct timeval tpEnd;
    gettimeofday(&tpEnd, NULL);

    long long multiple = 1000000ll;
    return tpEnd.tv_sec * multiple + tpEnd.tv_usec - (tpBegin.tv_sec * multiple + tpBegin.tv_usec);
}
#endif

#ifdef MULTICARD_ONLY_GPU
void partitionPos(int ig, int pos0, int pos1, int mode, int dev_id)
{
/*
	mode : 1: the sites are equally divided among all the GPU cards.
	       0: the sites are assigned to only one GPU specified by dev_id.
*/
	if(mode==1){
      int step = (pos1 - pos0) / dev_count;
	  thread_pos0[ig][0] = pos0;
      thread_pos1[ig][0] = thread_pos0[ig][0] + step;
	  int i;
      for (i = 1; i < dev_count; i++) {
          thread_pos0[ig][i] = thread_pos1[ig][i - 1];
          thread_pos1[ig][i] = thread_pos0[ig][i] + step;
      }
      thread_pos1[ig][dev_count - 1] = pos1;
	}
	else{
	  if(mode==0){
		if(dev_id >= dev_count){
		  printf("No such device!\n");
		  exit(-1);
		}
		int i;
	    for(i=0; i<dev_id; i++){
		  thread_pos0[ig][i] = pos0;
		  thread_pos1[ig][i] = pos0;
		}
		thread_pos0[ig][dev_id] = pos0;      /* only the device specified by dev_id does the calculation */
		thread_pos1[ig][dev_id] = pos1;
		for(i=dev_id+1; i<dev_count; i++){
		  thread_pos0[ig][i] = pos1;
		  thread_pos1[ig][i] = pos1;
		}
	  }
	  else{
	    printf("Error in mode!\n");
		exit(-1);
	  }
	}
}

void partitionTask(double *x)
{
/*
  Partition the tasks among GPUs if MULTICARD_ONLY_GPU is defined or among CPU and GPUs if HYBRID is defined.
*/
    printf("\nGPU Cards: %d\n", dev_count);
    printf("Partition result:\n");

    int ig;
    for (ig = 0; ig < com.ngene; ig++) {
        int pos0 = com.posG[ig];
        int pos1 = com.posG[ig + 1];

#ifdef DIVIDE_EQUALLY                                      
		partitionPos(ig, pos0, pos1, 1, 0);               /* divide the sites equally among all the GPUs*/
#else
		int dev_id;

		/* compare the performance of unequally dividing and equally dividing and choose the better one. */
		int count=30;
		long long equal_time = 0, unequal_time = 0;
		partitionPos(ig, pos0, pos1, 1, 0);                         /* equal partition */
		equal_time = posTest(ig, x, count);

		long long GPUtime[MAXGPU], total_time=0;
        for(dev_id=0; dev_id < dev_count; dev_id++){
		   GPUtime[dev_id] = 0;
           partitionPos(ig, pos0, pos1, 0, dev_id);                   /* assign all the sites to one GPU card to test its performance. */
           GPUtime[dev_id] = posTest(ig, x, count);                    /* record the time */
		   total_time += GPUtime[dev_id];
		}
        thread_pos0[ig][0] = pos0;
		thread_pos1[ig][0] = pos0 + double(GPUtime[0])/double(total_time)*(pos1-pos0);
		for(dev_id=1; dev_id < dev_count; dev_id++){
		  thread_pos0[ig][dev_id] = thread_pos1[ig][dev_id-1];
		  thread_pos1[ig][dev_id] = thread_pos0[ig][dev_id] + double(GPUtime[dev_id])/double(total_time)*(pos1-pos0);
		}
        thread_pos1[ig][dev_count-1] = pos1;
		unequal_time = posTest(ig, x, count);
        if(equal_time < unequal_time){
		  partitionPos(ig, pos0, pos1, 1, 0);
		}
#endif             /* DIVIDE_EQUALLY */
        printf("\tGene No.%d (pos %d to %d):\n", ig, pos0, pos1);
        int i;
        for (i = 0; i < dev_count; i++) {
            printf("\t\tGPU Card%d: %d %d\n", i, thread_pos0[ig][i], thread_pos1[ig][i]);
        }
    }
}
#else             /* MULTICARD_ONLY_GPU */

#ifdef HYBRID
void partitionPos(int ig, int pos0, int pos1, int cpu_pos)
{
/*
  The sites are assigned to GPUs and CPU.
*/
    int step = (pos1 - pos0 - cpu_pos) / dev_count;
    int i;
	thread_pos0[ig][0] = pos0;
	thread_pos1[ig][0] = pos0 + step;
    for (i = 1; i < dev_count; i++) {
        thread_pos0[ig][i] = thread_pos1[ig][i - 1];
        thread_pos1[ig][i] = thread_pos0[ig][i] + step;
    }
    thread_pos1[ig][dev_count - 1] = pos1 - cpu_pos;
    thread_pos0[ig][dev_count] = pos1 - cpu_pos;
    thread_pos1[ig][dev_count] = pos1;
}

void partitionTask(double *x)
{
    printf("\nGPU Cards: %d\n", dev_count);
    int num_procs = MAXCPU;
    printf("Total available threads: %d\n", num_procs);
    int gpu_procs = (dev_count + 1) / 2;                 // Approximately two cards use up one core.
    int cpu_procs = MAX(0, num_procs - gpu_procs);
    if (cpu_procs) {
        omp_set_num_threads(cpu_procs);
    }
    printf("Use %d threads for GPU and %d for CPU\n", gpu_procs, cpu_procs);
    printf("Partition result:\n");

    int ig;
    for (ig = 0; ig < com.ngene; ig++) {
        int pos0 = com.posG[ig];
        int pos1 = com.posG[ig + 1];

        int cpu_pos = 0;
        if (cpu_procs) {
            int count = 10;
            partitionPos(ig, pos0, pos1, 0);
            const long long zerotime = posTest(ig, x, count);
            partitionPos(ig, pos0, pos1, pos1);
            const long long fulltime = posTest(ig, x, count);

            const int testpos = pos0 + 1.0 / (1 + (double) fulltime / zerotime) * (pos1 - pos0);

            long long mintime = zerotime;
            int step = MAX(1, (pos1 - pos0) / 1000);
            int pos;
            for (pos = testpos - step; pos >= 0; pos -= step) {
                partitionPos(ig, pos0, pos1, pos);
                long long now = posTest(ig, x, count);
                if (now < mintime) {
                    mintime = now;
                    cpu_pos = pos;
                }
            }
        }
        partitionPos(ig, pos0, pos1, cpu_pos);

        printf("\tGene No.%d (pos %d to %d):\n", ig, pos0, pos1);
        int i;
        for (i = 0; i < dev_count; i++) {
            printf("\t\tGPU Card%d: %d %d\n", i, thread_pos0[ig][i], thread_pos1[ig][i]);
        }
        printf("\t\t      CPU: %d %d\n", thread_pos0[ig][dev_count], thread_pos1[ig][dev_count]);
    }
}
#endif                /* HYBRID */
#endif                /* MULTICARD_ONLY_GPU */

void CUDA_Init(double *x)
{
/*
  Initialize all devices and partition the tasks.
*/
  
#ifdef HYBRID_OR_MULTICARD
    cutilSafeCall(cudaGetDeviceCount(&dev_count));
    cutilCheckMsg("cudaGetDeviceCount failed");
    if (dev_count == 0) {
        fprintf(stderr, "No device available!\n");
        exit(-1);
    }
    dev_count = MIN(dev_count, MAXGPU);

    int dev_id;
    for (dev_id = 0; dev_id < dev_count; dev_id++) {
		deviceInit(dev_id);
    }
#else
    deviceInit(SINGLE_GPU_ID);
#endif

    memset(extendUVR, 0, 3 * UVR_SIZE * sizeof(double));

    prepareTree();
#ifdef HYBRID
    int i;
    for (i = 0; i < tree.nnode; i++) {
        nodes_conP[i] = (double *) malloc(sizeof(double) * 64 * com.npatt);
    }
#endif

#ifdef HYBRID_OR_MULTICARD
    partitionTask(x);                                  /* partition the task */
#endif
}

void deviceFreeAll(int dev_id)
{
/*
  Free the memory allocated before and destroy the stream created before.
*/
#ifdef HYBRID_OR_MULTICARD
    cutilSafeCall(cudaSetDevice(dev_id));
    cutilCheckMsg("cudaSetDevice failed");
#endif

#ifdef STREAM
    int i;
    for (i = 0; i < com.ns; i++) {
        cutilSafeCall(cudaStreamDestroy(stream[dev_id][i]));
        cutilCheckMsg("cudaStreamDestroy failed");
    }
#endif

    cutilSafeCall(cudaFree(dev_com_z[dev_id]));
    cutilCheckMsg("cudaFree dev_com_z failed");
    cutilSafeCall(cudaFree(dev_com_nodeScaleF[dev_id]));
    cutilCheckMsg("cudaFree dev_com_nodeScaleF failed");
    cutilSafeCall(cudaFree(dev_com_conP[dev_id]));
    cutilCheckMsg("cudaFree dev_com_conP failed");
    cutilSafeCall(cudaFree(dev_PMat[dev_id]));
    cutilCheckMsg("cudaFree dev_PMat failed");
    cutilSafeCall(cudaFree(dev_UVR[dev_id]));
    cutilCheckMsg("cudaFree dev_UVR failed");
    cutilSafeCall(cudaFree(dev_exptRootAll[dev_id]));
    cutilCheckMsg("cudaFree dev_exptRootAll failed");
    cutilSafeCall(cudaFree(dev_PMatT[dev_id]));
    cutilCheckMsg("cudaFree dev_PMatT failed");
    cutilSafeCall(cudaFree(dev_PMatiUVR[dev_id]));
    cutilCheckMsg("cudaFree dev_PMatiUVR failed");
}

void CUDA_FreeAll(void)
{
#ifdef HYBRID_OR_MULTICARD
    int dev_id;
    for (dev_id = 0; dev_id < dev_count; dev_id++) {
        deviceFreeAll(dev_id);
    }   
#ifdef HYBRID
	int i;
    for (i = 0; i < tree.nnode; i++) {
        free(nodes_conP[i]);
    }
#endif
#else                    /* HYBRID_OR_MULTICARD */
    deviceFreeAll(SINGLE_GPU_ID);
#endif
}

extern double _rateSite;
extern int IClass;
extern int NPMatUVRoot;
extern double Qfactor_NS_branch[NBTYPE];

void preparePMat(int n, int igene, double *x)
{
   assert(com.seqtype==CODONseq  && com.NSsites && com.model);
   assert(com.model<=NSbranch2); /* branch-site models A & B */
   assert(!(com.seqtype == AAseq && com.model == Poisson));

   int i;
   for (i = 0; i < tree.nnode; i++) {
       if (i == tree.root) continue;

       double t = nodes[i].branch * _rateSite;
       if(com.clock<5) {
           if(com.clock)  t *= GetBranchRate(igene, (int)nodes[i].label, x, NULL);
           else           t *= com.rgene[igene];
       }

       int iUVR=0, ib = (int)nodes[i].label;

       if(ib==0) iUVR = IClass%2;                  /* background, w0 w1 */
       else      iUVR = (IClass<=1 ? IClass : 2);  /* foreground, w0 w1 w2 */

       t *= Qfactor_NS_branch[ib];

       NPMatUVRoot++;

       if (t<-0.1) printf ("\nt = %.5f in PMatUVRoot", t);

       PMatT[i] = t;
       PMatiUVR[i] = iUVR;                      /* which set of U,V,R to use */
   }
}

void prepareUVR()
{
/*
  Extend the U,V matrix to 64*64, and R to 64.
*/
    double *UVR = PMat + 64 * 64;
    int i;
    for (i = 0; i < 3; i++) {
        int j;
        for (j = 0; j < com.ncode; j++) {
            memcpy(extendUVR + i * UVR_SIZE + j * 64, UVR + i * UVR_SIZE + j * com.ncode, com.ncode * sizeof(double));
        }
        for (j = 0; j < com.ncode; j++) {
            memcpy(extendUVR + i * UVR_SIZE + 64 * 64 + j * 64, UVR + i * UVR_SIZE + 64 * 64 + j * com.ncode, com.ncode * sizeof(double));
        }
        memcpy(extendUVR + i * UVR_SIZE + 64 * 64 * 2, UVR + i * UVR_SIZE + 64 * 64 * 2, com.ncode * sizeof(double));
    }
}

void nodeScale(int inode, int dev_id, int pos0, int pos1)
{
/* 
  Scale to avoid underflow. 
*/
    callKernelNodeScale(inode, dev_id, pos0, pos1, nodeScaleK[inode]);
}

#ifdef HYBRID
void *calcPMatAll(void *arg)
{
/*
  If HYBRID is defined, calculate all the transition matrix on device 0.
*/
    cutilSafeCall(cudaSetDevice(0));
    cutilCheckMsg("cudaSetDevice failed");

    callKernelPMatUVRootAll(0);

    cutilSafeCall(cudaMemcpy(PMatAll, dev_PMat[0], tree.nnode * 64 * 64 * sizeof(double), cudaMemcpyDeviceToHost));
    cutilCheckMsg("cudaMemcpy PMatAll failed");

    return NULL;
}
#endif

#ifdef HYBRID_OR_MULTICARD
void *schedule(void *arg)
#else
void schedule(int dev_id, int pos0, int pos1)
#endif
{
/*
  Schedule the calculation on GPU, first the transition matrix, then conditional probability, if needed, scale the contitional probability.
*/
#ifdef HYBRID_OR_MULTICARD
    int dev_id = (long) arg;
    int pos0 = thread_pos0[thread_ig][dev_id];
    int pos1 = thread_pos1[thread_ig][dev_id];
    if (pos0 == pos1) return NULL;

    cutilSafeCall(cudaSetDevice(dev_id));
    cutilCheckMsg("cudaSetDevice failed");
#endif

#ifdef HYBRID
    if (dev_id != 0) callKernelPMatUVRootAll(dev_id);
#else
    callKernelPMatUVRootAll(dev_id);                   /* if HYBRID is not defined, each GPU calculates transition matrix once for later use */
#endif

    int i;
    for (i = 0; i < tree.nnode + com.NnodeScale - 1; i++) {
        int inode = schedule_que[i];
        if (schedule_scale[i]) {                        /* the current node needs scaling */
            nodeScale(inode, dev_id, pos0, pos1);
        } else if (nodes[inode].nson < 1 && com.cleandata) {
            callKernelConditionalPNodeTipClean(dev_id, pos0, pos1, father[inode], inode);
        } else if (nodes[inode].nson < 1 && !com.cleandata) {
            callKernelConditionalPNodeTipUnclean(dev_id, pos0, pos1, father[inode], inode);
        } else {             /* internal */
#ifdef STREAM
            cutilSafeCall(cudaStreamSynchronize(stream[dev_id][inode - com.ns]));
            cutilCheckMsg("cudaStreamSynchronize failed");
#endif
            callKernelConditionalPNodeInternal(dev_id, pos0, pos1, father[inode], inode);
        }
    }

    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg("cudaDeviceSynchronize failed");            
	
    callKernelCalcFhK(dev_id, pos0, pos1);

    cutilSafeCall(cudaMemcpy(com.fhK + global_ir *com.npatt + pos0, dev_com_fhK[dev_id] + pos0, (pos1 - pos0) * sizeof(double), cudaMemcpyDeviceToHost));
    cutilCheckMsg("cudaMemcpy com_fhK failed");

#ifdef HYBRID_OR_MULTICARD
    return NULL;
#endif
}

void CUDA_ScheduleConditionalPNode(int ir, int ig, double *x)
{
/*
  If HYBRID is defined, create one thread for each device and schedule the calculation on all devices, otherwise do all the calculation on the single device.
*/
    assert(com.ncode >= 61);

    preparePMat(com.ncode, ig, x);
    prepareUVR();

    global_ir = ir;

#ifdef MULTICARD_ONLY_GPU
	thread_ig = ig;
	int dev_id;
    for (dev_id = 0; dev_id < dev_count; dev_id++) {
        if (pthread_create(&threads[dev_id], NULL, schedule, (void *) dev_id) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
	for (dev_id = 0; dev_id < dev_count; dev_id++) {
        if (pthread_join(threads[dev_id], NULL) != 0) {
            perror("pthread_join");
            exit(EXIT_FAILURE);
        }
    }
#else             /* MULTICARD_ONLY_GPU */
#ifdef HYBRID
    thread_ig = ig;

    if (pthread_create(&threads[0], NULL, calcPMatAll, NULL) != 0) {
        perror("pthread_create");
        exit(EXIT_FAILURE);
    }
    int dev_id;
    for (dev_id = 1; dev_id < dev_count; dev_id++) {
        if (pthread_create(&threads[dev_id], NULL, schedule, (void *) dev_id) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
    if (pthread_join(threads[0], NULL) != 0) {
        perror("pthread_join");
        exit(EXIT_FAILURE);
    }
    if (pthread_create(&threads[0], NULL, schedule, (void *) 0) != 0) {
        perror("pthread_create");
        exit(EXIT_FAILURE);
    }

    int pos0 = thread_pos0[thread_ig][dev_count];
    int pos1 = thread_pos1[thread_ig][dev_count];
    if (pos0 != pos1) {
        OMP_CalcFhk(com.fhK + global_ir * com.npatt, pos0, pos1);                     
    }

    for (dev_id = 0; dev_id < dev_count; dev_id++) {
        if (pthread_join(threads[dev_id], NULL) != 0) {
            perror("pthread_join");
            exit(EXIT_FAILURE);
        }
    }
#else                       /* HYBRID */
    int pos0 = com.posG[ig];
    int pos1 = com.posG[ig + 1];

    schedule(SINGLE_GPU_ID, pos0, pos1);
#endif                   /* HYBRID */
#endif                   /* MULTICARD_ONLY_GPU */
}

