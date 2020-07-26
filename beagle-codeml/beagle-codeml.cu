#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <omp.h>
#include <sys/time.h>
#include "paml_beagle.h"
#include "codeml_beagle.h"
#ifdef BEAGLE
#include "/home/hxm/beagle_debug_sync/include/libhmsbeagle-1/libhmsbeagle/beagle.h"
#endif


#ifdef BEAGLE
  void Traverse(int inode);
  int GetOperations(void);
  double * GetTipPartials(int igene, int inode);
  void BEAGLE_Alloc(void);
  void BEAGLE_free();
  void BEAGLE_CreateInstance();
  void BEAGLE_InitValues();
  void BEAGLE_SetValues();
  void BEAGLE_InitAll();
  void BEAGLE_SetCategoryWeights();
  void BEAGLE_SetTransitionMatrix(int ig, double x[]);
  void BEAGLE_UpdatePartials();
  void BEAGLE_AccumulateScaleFactors();
  double BEAGLE_CalculateRootLogLikelihoods();
  void BEAGLE_Finish(); 
  double BEAGLE_ScheduleAll(int ig, double x[]);

#ifdef CUDA
  void preparePMat(int n, int igene, double *x);
  void prepareUVR();
#endif
#endif                    /* BEAGLE */


#ifdef BEAGLE
#define IndexSize 3*500
double * beagle_PMat;
double * PMat_temp;
int * scalingFactorsIndices ;
double * partials;
double * rates;
double * paddedValues ;
int * nodeIndices ;
double * PatternWeights;
int * PartialsIndex;
int * TransMatrixIndex;
int instance;
int opCnt;
BeagleOperation * operations ;
int cumulativeScalingIndex, scalingFactorsCount;

#ifdef CUDA
__constant__ int DEV_tree_root_beagle;
__constant__ int DEV_com_ncode_beagle;

double PMatT_beagle[NNODE];
int PMatiUVR_beagle[NNODE];
double __attribute((aligned(0x10))) extendUVR_beagle[3 * UVR_SIZE];

double *dev_PMat_beagle;		// tree.nnode * (tree.ncode * tree.ncode)
double *dev_UVR_beagle;
double *dev_exptRootAll_beagle;
double *dev_PMatT_beagle;
int *dev_PMatiUVR_beagle;
extern double _rateSite;
extern int IClass;
extern int NPMatUVRoot;
extern double Qfactor_NS_branch[NBTYPE];


void preparePMat(int n, int igene, double *x)
{
/*
  Prepare PMatT_beagle[] and PMatiUVR_beagle[].(From cuda-codeml.cu)
*/
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

       PMatT_beagle[i] = t;
       PMatiUVR_beagle[i] = iUVR;
   }
}


void prepareUVR()
{
/*
  Extend U,V to 64*64, R to 64.(From cuda-codeml.cu)
*/
    double *UVR = PMat + 64 * 64;
    int i;
    for (i = 0; i < 3; i++) {
        int j;
        for (j = 0; j < com.ncode; j++) {
            memcpy(extendUVR_beagle + i * UVR_SIZE + j * 64, UVR + i * UVR_SIZE + j * com.ncode, com.ncode * sizeof(double));
        }
        for (j = 0; j < com.ncode; j++) {
            memcpy(extendUVR_beagle + i * UVR_SIZE + 64 * 64 + j * 64, UVR + i * UVR_SIZE + 64 * 64 + j * com.ncode, com.ncode * sizeof(double));
        }
        memcpy(extendUVR_beagle + i * UVR_SIZE + 64 * 64 * 2, UVR + i * UVR_SIZE + 64 * 64 * 2, com.ncode * sizeof(double));
    }
}


void deviceInit_beagle(int dev_id)
{
/*
  Initialize the device specified by dev_id. (From cuda-codeml.cu)
*/
  cudaSetDevice(dev_id);
  cudaMemcpyToSymbol(DEV_tree_root_beagle, &tree.root, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(DEV_com_ncode_beagle, &com.ncode, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&dev_PMat_beagle, tree.nnode * 64 * 64 * sizeof(double));
  cudaMalloc((void **)&dev_UVR_beagle, 3 * (64 * 64 * 2 + 64) * sizeof(double));
  cudaMalloc((void **)&dev_exptRootAll_beagle, tree.nnode * 64 * sizeof(double));
  cudaMalloc((void **)&dev_PMatT_beagle, tree.nnode * sizeof(double));
  cudaMalloc((void **)&dev_PMatiUVR_beagle, tree.nnode * sizeof(int));
  //memset(extendUVR_beagle, 0, 3 * UVR_SIZE * sizeof(double));
}


__global__
void kernelPMatExptRoot_beagle(const double *PMatT, const int *PMatiUVR, const double *UVR, double *exptRootAll)
{
/*
  Kernel of calculating exptRootAll[]=exp(t*UVR[]).(From cuda-codeml.cu)
*/
    int inode = blockIdx.x;
    if (inode == DEV_tree_root_beagle) return ;

    int tx = threadIdx.x;
    int idx = inode * 64 + tx;
    double t = PMatT[inode];
    int ridx = PMatiUVR[inode] * UVR_SIZE + 64 * 64 * 2 + tx;
    exptRootAll[idx] = exp(t * UVR[ridx]);
}


 __device__
inline void saxpy_beagle(double a, const double *b, double *c)
{
#pragma unroll
    for (int i = 0; i < 16; i++) {
        c[i] += a * b[i];
    }
}


__global__
void kernelPMatUVRoot_beagle(const int *PMatiUVR, const double *UVR, const double *exptRootAll, double *PMat)
{
/*
  Calculate transition matrix. (From cuda-codeml.cu)
*/
    const int inode = blockIdx.y;
    if (inode == DEV_tree_root_beagle) return ;

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
    if (id < DEV_com_ncode_beagle) exptRoot = exptRootAll[inode * 64 + id];

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
        saxpy_beagle(v0, bs[i], c);
    }

#pragma unroll
    for (int i = 0; i < m; i++) {
        if (c[i] < 0) c[i] = 0;
        P[i] = c[i];
    }
}


void callKernelPMatUVRootAll( int ir)
{   
    cudaMemcpy(dev_UVR_beagle, extendUVR_beagle, 3 * UVR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_PMatT_beagle, PMatT_beagle, tree.nnode * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_PMatiUVR_beagle, PMatiUVR_beagle, tree.nnode * sizeof(int), cudaMemcpyHostToDevice);

    kernelPMatExptRoot_beagle<<<tree.nnode, com.ncode>>>(dev_PMatT_beagle, dev_PMatiUVR_beagle, dev_UVR_beagle, dev_exptRootAll_beagle);

    dim3 threadsPerBlock(64);
    dim3 blocksPerGrid(4, tree.nnode);
    kernelPMatUVRoot_beagle<<<blocksPerGrid, threadsPerBlock>>>(dev_PMatiUVR_beagle, dev_UVR_beagle, dev_exptRootAll_beagle, dev_PMat_beagle);
	cudaMemcpy(PMat_temp , dev_PMat_beagle, 64 * 64 * tree.nnode * sizeof(double), cudaMemcpyDeviceToHost);

	int i,j,k;
	for(i = 0; i < tree.nnode; i++){
		if(i == tree.root){
		 for(j = 0; j < com.ncode; j++){
	      for(k = 0; k < com.ncode; k++){
		   beagle_PMat[i * com.ncatG * com.ncode * com.ncode + ir * com.ncode * com.ncode + j * com.ncode + k] = 0;
		  }
		  beagle_PMat[i * com.ncatG * com.ncode * com.ncode + ir * com.ncode * com.ncode + j * com.ncode + j] = 1;
		 }
		 continue;
		}
		for(j = 0; j < com.ncode; j++){
	     for(k = 0; k < com.ncode; k++){
		   beagle_PMat[i * com.ncatG * com.ncode * com.ncode + ir * com.ncode * com.ncode + j * com.ncode + k] = PMat_temp[i * 64 * 64 + k * 64 + j];
		 }
		}
	}
}
#endif              /* CUDA */


 void Traverse(int inode)
{
/*
  Traverse the tree(print the sons of each internal node), used for testing.
*/
	 if(nodes[inode].nson > 0){
	   printf("sons of node %d :\n",inode);
	   int i;
	   for(i = 0; i < nodes[inode].nson; i++){
	     printf("%d ,",nodes[inode].sons[i]);
	   }
	   printf("\n");
	   int j;
	   for(j = 0; j < nodes[inode].nson; j++){
	     Traverse(nodes[inode].sons[j]);
	   }
	 }
	 return ;
}


int GetOperations(void)
{
/*
  Arrange operations of updatePartials. Every three elements are used for one operation: index of child1, index of child2, index of parent. 
*/
	memset(PartialsIndex, 0 ,sizeof(int) * IndexSize);
	memset(TransMatrixIndex, 0 ,sizeof(int) * IndexSize);
	int cnt = 0;
	int i, j = 0, k = 0;
	for(i = tree.nnode-1; i >= tree.root; i--){
	  int son0 = nodes[i].sons[0];
	  int son1 = nodes[i].sons[1];
	  PartialsIndex[j++] = son0;
	  PartialsIndex[j++] = son1;
	  PartialsIndex[j++] = i;
	  TransMatrixIndex[k++] = son0;
	  TransMatrixIndex[k++] = son1;
	  cnt++;
	}
	//If the tree is un unrooted tree, one more node is assumed at the root.
	if(nodes[tree.root].nson > 2){
	  PartialsIndex[j++] = tree.root;
	  PartialsIndex[j++] = nodes[tree.root].sons[2];
	  PartialsIndex[j++] = tree.nnode;
	  TransMatrixIndex[k++] = tree.root;
	  TransMatrixIndex[k++] = nodes[tree.root].sons[2];
	  cnt++;
	}

	if((j%3 != 0) || (k%2 != 0)){
	  return -1;
	}
	return cnt;
}


double * GetTipPartials(int igene,int inode)
{
/*
  Get partials at the node inode.
*/
  int h, pos0 = com.posG[igene], pos1 = com.posG[igene+1], n = com.ncode, k;
  double *tmp_partials = (double*)malloc(sizeof(double) * com.npatt * n);
  memset(tmp_partials, 0, sizeof(double) * com.npatt * n);
     if(com.cleandata){
      for(h = pos0; h < pos1; h++){
         tmp_partials[(h-pos0) * n + com.z[inode][h-pos0]] = 1.0; 
	  }
	 }
	 else{
	  for(h = pos0; h < pos1; h++){
	    for(k = 0;k < nChara[com.z[inode][h-pos0]]; k++){
		  tmp_partials[(h-pos0) * n + CharaMap[com.z[inode][h-pos0]][k]] = 1.0;
		}
	  }
	 }
  return tmp_partials;
}


void BEAGLE_Alloc(void)
{
/*
  Allocate memory for arrays.
*/
     beagle_PMat = (double*)malloc(sizeof(double) * com.ncode * com.ncode * tree.nnode * com.ncatG);       //each node has a transition matrix including the root(In)
	 PMat_temp=(double*)malloc(sizeof(double) * tree.nnode * 64 * 64);
#ifdef NOSCALE
#else
	 if(com.NnodeScale != 0){
	 scalingFactorsIndices = (int*)malloc(sizeof(int) * com.NnodeScale);
	 }
#endif
	 partials = (double*)malloc(sizeof(double) * com.ncode * com.npatt);
	 rates = (double*)malloc(sizeof(double) * com.ncatG);
	 paddedValues = (double*)malloc(sizeof(double) * 64);
	 nodeIndices = (int*)malloc(sizeof(int) * tree.nnode);
	 PatternWeights = (double*)malloc(sizeof(double) * com.npatt);
	 PartialsIndex = (int*)malloc(sizeof(int) * IndexSize);
	 TransMatrixIndex = (int*)malloc(sizeof(int) * IndexSize);
}


void BEAGLE_free()
{
/*
  Free the memory allocated before.
*/
	if(beagle_PMat){
      free(beagle_PMat);
	}
	if(PMat_temp){
	  free(PMat_temp);
	}
	if(scalingFactorsIndices){
     free(scalingFactorsIndices);
	}
	if(partials){
     free(partials);
	}
	if(rates){
     free(rates);
	}
	if(paddedValues){
     free(paddedValues);
	}
	if(nodeIndices){
     free(nodeIndices);
	}
	if(PatternWeights){
	  free(PatternWeights);
	}
	if(PartialsIndex){
	  free(PartialsIndex);
	}
	if(TransMatrixIndex){
	  free(TransMatrixIndex);
	}
  return;
}


void BEAGLE_CreateInstance()
{
/*
 Create an instance.
*/	
#ifdef NOSCALE
   int scaleCount = 0;
#else
   int scaleCount = ((com.NnodeScale==0)? 0 : com.NnodeScale+1);
#endif

   int Nnode=(nodes[tree.root].nson>2 ? tree.nnode+1 : tree.nnode);                 //The number of nodes of the tree. If the tree is an unrooted tree, then Nnode will be tree.nnode+1

   BeagleInstanceDetails instDetails;
   
   int dev_count;
   cudaGetDeviceCount(&dev_count);
   if(dev_count == 0){
     printf("No device available!\n");
	 exit(1);
   }
   int gpu_id = BEAGLE_GPU_ID;
   if(gpu_id > dev_count||gpu_id <= 0){
	  printf("Error! BEAGLE_GPU_ID set wrong (should be 1 to #devices).\n");
	  exit(1);
   }
   int res_length = 1;	
   instance = beagleCreateInstance(
                                  com.ns,			   // Number of tip data elements (input) 
                                  Nnode,	           // Number of partials buffers to create (input) 
                                  0,		           // Number of compact state representation buffers to create (input) 
                                  com.ncode,		   // Number of states in the continuous-time Markov chain (input) 
                                  com.npatt,		   // Number of site patterns to be handled by the instance (input) 
                                  1,		           // Number of rate matrix eigen-decomposition buffers to allocate (input)
                                  tree.nnode,		   // Number of rate matrix buffers (input) 
                                  com.ncatG,           // Number of rate categories (input)
                                  scaleCount,          // Number of scaling buffers 
                                  &gpu_id,//NULL,       // List of potential resource on which this instance is allowed (input, NULL implies no restriction 
                                  res_length, //0, 		    // Length of resourceList list (input)
                                  BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_SCALING_MANUAL ,             	// Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) 
                                  0 | BEAGLE_FLAG_PRECISION_DOUBLE | BEAGLE_FLAG_FRAMEWORK_CUDA | BEAGLE_FLAG_SCALERS_LOG,           // Bit-flags indicating required implementation characteristics, see BeagleFlags (input) 
                                  &instDetails);
    if (instance < 0) {
	    printf("Failed to obtain BEAGLE instance\n\n");
		BEAGLE_Finish();
	    exit(1);
    }

  return ;
}


void BEAGLE_InitValues()
{
/*
  Initialize relevant values.
*/
    int i;

    //set nodeIncies for transition matrix.
    for(i = 0; i < tree.nnode; i++){
	  nodeIndices[i] = i;
	}

	//set padded values for state
	for(i = 0; i < 64; i++){                
	  paddedValues[i] = 0.0;
	}

	//set rates 
	for(i = 0; i < com.ncatG; i++) {
        rates[i] = 1.0;
    }

	 //set pattern weight    
	for(i=0; i < com.npatt; i++){
	  PatternWeights[i] = com.fpatt[i];
	}

	//init cumulativeScalingIndex(index of the cumulated scaling factors) and scalingFactorsIndices(index of scaling factors in scalingBuffer)
#ifdef NOSCALE
	cumulativeScalingIndex = BEAGLE_OP_NONE;
#else
	  if(com.NnodeScale > 0){
	    cumulativeScalingIndex = com.NnodeScale;
        scalingFactorsCount = com.NnodeScale;
		int fcnt = 0, curnode;
		for(curnode = 0; curnode < com.NnodeScale; curnode++){
		  scalingFactorsIndices[fcnt++] = curnode;
		}
		
		if(curnode != com.NnodeScale){
		  printf("Error ! curnode!= com.NnodeScale\n");
		  exit(1);
		}
	  }
#endif                               /* NOSCALE */
	return ;
}


void BEAGLE_SetValues()
{
/*
  Set values.
*/
  int retValue;

  //set rate categories 
  retValue = beagleSetCategoryRates(instance, &rates[0]);    
  if(retValue < 0){
	  printf("Error in beagleSetCategoryRates! Error code is : %d\n", retValue);
	  BEAGLE_Finish();
	  exit(1);
  }

     //set pattern weights 
   retValue = beagleSetPatternWeights(instance, PatternWeights);
   if(retValue < 0){
	  printf("Error in beagleSetPatternWeights! Error code is : %d\n", retValue);
	  BEAGLE_Finish();
	  exit(1);
   }
	
	//set base frequence of codon
	retValue = beagleSetStateFrequencies(instance, 0, com.pi);
	if(retValue < 0){
	   printf("Error in beagleSetStateFrequencies! Error code is : %d\n", retValue);
	   BEAGLE_Finish();
	   exit(1);
	}
	
	int i;
	//set tip partials
	for(i=0; i < com.ns; i++){
	  partials = GetTipPartials(0, i);
	  // set the sequences for each tip using partial likelihood arrays
	  retValue = beagleSetTipPartials(instance, i, partials);
	  if(retValue < 0){
	   printf("Error in beagleSetTipPartials of node %d! Error code is : %d\n", i, retValue);
	   BEAGLE_Finish();
	   exit(1);
	  }
	}
  
    //  Create a list of partial likelihood update operations, the order is [dest, destScaling, source1, matrix1, source2, matrix2]
	opCnt = GetOperations();
	if(opCnt <= 0){
	  printf("Error in GetOperations!\n");
	  BEAGLE_Finish();
	  exit(1);
	}
	operations = (BeagleOperation *)malloc(sizeof(BeagleOperation)*(opCnt));
	int scalecnt = 0;
	int parI = 0, matI = 0;
	for(i = 0; i < opCnt; i++){
		operations[i].child1Partials = PartialsIndex[parI++];
		operations[i].child2Partials = PartialsIndex[parI++];
		operations[i].destinationPartials = PartialsIndex[parI++];
		operations[i].child1TransitionMatrix = TransMatrixIndex[matI++];
		operations[i].child2TransitionMatrix = TransMatrixIndex[matI++];
		operations[i].destinationScaleRead = BEAGLE_OP_NONE;
		
#ifdef NOSCALE
		operations[i].destinationScaleWrite = BEAGLE_OP_NONE;
#else
        if(operations[i].destinationPartials == tree.nnode){
		  operations[i].destinationScaleWrite = BEAGLE_OP_NONE;
		  continue;
		}
		if(operations[i].destinationPartials != tree.root && com.NnodeScale && com.nodeScale[operations[i].destinationPartials]){
			operations[i].destinationScaleWrite = scalecnt;
			scalecnt++;
		}
		else{
			operations[i].destinationScaleWrite = BEAGLE_OP_NONE;
		}
#endif
	}
#ifdef NOSCALE
#else
	if(scalecnt != com.NnodeScale){
		printf("Error ! scalecnt!= com.NnodeScale\n");
		BEAGLE_Finish();
		exit(1);
	}
#endif
}


void BEAGLE_SetCategoryWeights()
{
/*
  Set rate catogery weights.
*/
    int retValue = beagleSetCategoryWeights(instance, 0, com.freqK);
	if(retValue < 0){
	   printf("Error in beagleSetCategoryWeights! Error code is : %d\n", retValue);
	   BEAGLE_Finish();
	   exit(1);
	}
}


void BEAGLE_InitAll(){

  BEAGLE_Alloc();                  /* allocate memory */

  BEAGLE_InitValues();             /* init values */

  BEAGLE_CreateInstance();        /* create an instance */

  BEAGLE_SetValues();

  int dev_id = 0;
  dev_id = BEAGLE_GPU_ID;
  deviceInit_beagle(dev_id);
}


void BEAGLE_SetTransitionMatrix(int ig, double x[])
{
/*
  Calculate the transition matrix and set the transition matrix for beagle.
*/
	/*
	double t;
	int j,k,i,ir;
	for(i=0;i<tree.nnode;i++){
	if(i==tree.root){
		  for(ir=0;ir<com.ncatG; ir++){
	       for(j=0;j<com.ncode;j++){
		    for(k=0;k<com.ncode;k++){
			  beagle_PMat[tree.root*com.ncode*com.ncode*com.ncatG + ir*com.ncode*com.ncode + j*com.ncode + k]=0;
			}
			beagle_PMat[tree.root*com.ncode*com.ncode*com.ncatG + ir*com.ncode*com.ncode + j*com.ncode + j]=1;
		   }
	     }
	}
	else{
	  for(ir=0;ir<com.ncatG; ir++){
	    SetPSiteClass(ir,x); 
		t = nodes[i].branch * _rateSite;
        if(com.clock<5) {
         if(com.clock)  t *= GetBranchRate(ig,(int)nodes[i].label,x,NULL);
         else           t *= com.rgene[ig];
          }
		  GetPMatBranch(PMat_temp, x, t, i);
		  for(j=0;j<com.ncode;j++){
		    for(k=0;k<com.ncode;k++){
			  beagle_PMat[i*com.ncode*com.ncode*com.ncatG + ir*com.ncode*com.ncode + j*com.ncode + k]=PMat_temp[j*com.ncode + k];
			}
		  }
	  }
	}
 }
 */
	memset(beagle_PMat, 0, sizeof(double) * com.ncode * com.ncode * tree.nnode * com.ncatG);
	memset(extendUVR_beagle, 0, 3 * UVR_SIZE * sizeof(double));
    int ir;
	//int dev_id = SINGLE_GPU_ID;
    for(ir = 0; ir < com.ncatG; ir++){
	  SetPSiteClass(ir, x); 
	  preparePMat(com.ncode, ig, x);
      prepareUVR();
	  callKernelPMatUVRootAll(ir);
	}

     int retValue = beagleSetTransitionMatrices(instance,
                                nodeIndices,
                                beagle_PMat,
                                paddedValues,
                                tree.nnode);
	 if(retValue < 0){
	   printf("Error in beagleSetTransitionMatrices! Error code is : %d\n", retValue);
	   BEAGLE_Finish();
	   exit(1);
	  }
}


void BEAGLE_UpdatePartials()
{
/*
  Update partials from the tip to the root.
*/
	int retValue = beagleUpdatePartials(instance,     // instance
                   operations,                        // eigenIndex
                   opCnt,                             // operationCount
                   BEAGLE_OP_NONE);                   // cumulative scaling index
	if(retValue < 0){
	   printf("Error in beagleUpdatePartials! Error code is : %d\n", retValue);
	   BEAGLE_Finish();
	   exit(1);
	}
}


void BEAGLE_AccumulateScaleFactors()
{
/*
  Accumulate all the scaling factors. 
*/
  int cumulativeScalingIndex, scalingFactorsCount;
#ifdef NOSCALE
	cumulativeScalingIndex = BEAGLE_OP_NONE;
#else
	  if(com.NnodeScale > 0){
	    cumulativeScalingIndex = com.NnodeScale;
        scalingFactorsCount = com.NnodeScale;
		int fcnt = 0, curnode;
		for(curnode = 0; curnode < com.NnodeScale; curnode++){
		  scalingFactorsIndices[fcnt++] = curnode;
		}

		if(fcnt != com.NnodeScale){
		  printf("Error ! fcnt!= com.NnodeScale\n");
		  BEAGLE_Finish();
		  exit(1);
		}
      
		int retValue = beagleResetScaleFactors(instance,
                                cumulativeScalingIndex);    //set the value of scaleFactor[cumulativeScalingIndex] to be 0 
        if(retValue < 0){
	      printf("Error in beagleResetScaleFactors! Error code is : %d\n",retValue);
		  BEAGLE_Finish();
	      exit(1);
	    }

        retValue = beagleAccumulateScaleFactors(instance,
                                     scalingFactorsIndices,
                                     scalingFactorsCount,
                                     cumulativeScalingIndex);     
		if(retValue < 0){
	      printf("Error in beagleAccumulateScaleFactors! Error code is : %d\n",retValue);
		  BEAGLE_Finish();
	      exit(1);
	    }
	}
#endif                     /* NOSCALE */
}


double BEAGLE_CalculateRootLogLikelihoods()
{
/*
  Calculate log likelihood of the root.
*/
    int categoryWeightsIndex = 0, stateFrequencyIndex = 0, cumulativeScalingIndex;
    double logL = 0.0;    
  
#ifdef NOSCALE
	cumulativeScalingIndex = BEAGLE_OP_NONE; 
#else
	if(com.NnodeScale == 0){
	  cumulativeScalingIndex = BEAGLE_OP_NONE; 
	}
	else{
	  cumulativeScalingIndex = com.NnodeScale;
	}
#endif
    // calculate the site likelihoods at the root node
	int Root = tree.root;
	int Nnode = (nodes[tree.root].nson>2 ? tree.nnode+1 : tree.nnode);
	if(Nnode > tree.nnode){
	  Root = Nnode-1;
	}

	int retValue = beagleCalculateRootLogLikelihoods(instance,           // instance
	                            (const int *)&Root,                      // bufferIndices
                                  &categoryWeightsIndex,                 // weights
                                  &stateFrequencyIndex,                  // stateFrequencies
                                &cumulativeScalingIndex,                 // cumulative scaling index
	                            1,                                       // count
	                            &logL);          
	  if(retValue < 0){
	      printf("Error in beagleCalculateRootLogLikelihoods! Error code is : %d\n", retValue);
		  BEAGLE_Finish();
	      exit(1);
	  }
	  return logL;
}


double BEAGLE_ScheduleAll(int ig, double x[])
{
   // set rate category weights 
    BEAGLE_SetCategoryWeights();

	//Set transition matrix for beagle.
	BEAGLE_SetTransitionMatrix(ig, x);

    //Update the partials
    BEAGLE_UpdatePartials();
	
	//Accumulate scaling factors
	BEAGLE_AccumulateScaleFactors();

	double logL = BEAGLE_CalculateRootLogLikelihoods();
	return logL;
}


void BEAGLE_Finish()
{
/*
  Destroy the instance, free the memory.
*/
  beagleFinalizeInstance(instance);
  BEAGLE_free();
}
#endif                      /* BEAGLE */
