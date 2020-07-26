#ifndef CODEML_H
#define CODEML_H

#define NS            700
#define NBRANCH       (NS*2-2)
#define NNODE         (NS*2-1)
#define MAXNSONS      100
#define NGENE         2000
#define LSPNAME       50
#define NCODE         64
#define NCATG         40
#define NBTYPE        8

#define NP            (NBRANCH*2+NGENE-1+2+NCODE+2)
/*
#define NP            (NBRANCH+NGENE-1+189+2+NCODE+2)
*/

#define MAXCARD 16
#define UVR_SIZE (64 * 64 * 2 + 64)

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

struct common_info {
   unsigned char *z[NS];
   char *spname[NS], seqf[512],outf[512],treef[512],daafile[512], cleandata;
   char oldconP[NNODE];       /* update conP for nodes? to save computation */
   int seqtype, ns, ls, ngene, posG[NGENE+1], lgene[NGENE], npatt,*pose, readpattern;
   int runmode,clock, verbose,print, codonf,aaDist,model,NSsites;
   int nOmega, nbtype, nOmegaType;  /* branch partition, AA pair (w) partition */
   int method, icode, ncode, Mgene, ndata, bootstrap;
   int fix_rgene,fix_kappa,fix_omega,fix_alpha,fix_rho,nparK,fix_blength,getSE;
   int np, ntime, nrgene, nkappa, npi, nrate, nalpha, ncatG, hkyREV;
   size_t sconP, sspace;
   double *fpatt, *space, kappa,omega,alpha,rho,rgene[NGENE], TipDate, TipDate_TimeUnit;
   double pi[NCODE], piG[NGENE][64], fb61[64];
   double f3x4[NGENE][12], *pf3x4, piAA[20];
   double freqK[NCATG], rK[NCATG], MK[NCATG*NCATG],daa[20*20], *conP, *fhK;
   double (*plfun)(double x[],int np);
   double hyperpar[4]; /* kostas, the hyperparameters for the prior distribution of distance & omega */
   double omega_fix;  /* fix the last w in the NSbranchB, NSbranch2 models 
          for lineages.  Useful for testing whether w>1 for some lineages. */
   int     conPSiteClass; /* conPSiteClass=0 if (method==0) and =1 if (method==1)?? */
   int     NnodeScale;
   char   *nodeScale;        /* nScale[ns-1] for interior nodes */
   double *nodeScaleF;       /* nScaleF[npatt] for scale factors */
  /* pomega & pkappa are used to communicate between SetParameters & ConditionalPNode 
     & eigenQcodon.  Try to remove them? */
   double *pomega, pkappa[5], *ppi;
};

struct TREEB {
   int  nbranch, nnode, root, branches[NBRANCH][2];
   double lnL;
};

struct TREEN {
   int father, nson, sons[MAXNSONS], ibranch, ipop;
   double branch, age, omega, *conP, label;
   char *nodeStr, fossil, usefossil;
};

enum NSBranchModels {NSbranchB=1, NSbranch2, NSbranch3};
enum AAModel {Poisson, EqualInput, Empirical, Empirical_F, FromCodon=6, REVaa_0=8, REVaa=9};

extern struct common_info com;
extern struct TREEB tree;
extern struct TREEN *nodes;
extern char nChara[256], CharaMap[256][64];
extern double *PMat;

#ifdef CUDA
#ifndef BEAGLE
extern double PMatAll[NNODE][64 * 64];
extern double PMatT[NNODE];
extern int PMatiUVR[NNODE];
extern double extendUVR[3 * UVR_SIZE];

extern int bfs_que[NS];
extern int nodeScaleK[NNODE];

#ifdef HYBRID
extern double *nodes_conP[NNODE];
#endif // HYBRID

#else        //BEAGLE
#define IndexSize 3*500
extern double * beagle_PMat;
extern double * PMat_temp;
extern int * scalingFactorsIndices ;
extern double * partials;
extern double * rates;
extern double * paddedValues ;
extern int * nodeIndices ;
extern double * PatternWeights;
extern int * PartialsIndex;
extern int * TransMatrixIndex;
extern int instance;
extern int opCnt;
extern int ITR;
extern int cumulativeScalingIndex,scalingFactorsCount;
extern long long total_PMat, total_conP;
#endif       //BEAGLE
#endif // CUDA


#ifdef __cplusplus
extern "C" {
#endif

#ifdef CUDA
  double CUDA_lfundG (double x[], int np);

#ifndef BEAGLE
  void CUDA_Init(double *);
  void CUDA_FreeAll(void);
  void CUDA_ScheduleConditionalPNode(int, int, double *);
  
  int CUDA_fx_r(double x[], int np);
  void OMP_CalcFhk(double *, int, int);
#else            //BEAGLE
  void Traverse(int inode);
  int GetOperations(void);
  double * GetTipPartials(int igene,int inode);
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
  double BEAGLE_fx_r(double x[], int np);
  void BEAGLE_Finish(); 
  double BEAGLE_ScheduleAll(int ig, double x[]);
#endif              //BEAGLE
#endif              //CUDA

  double GetBranchRate(int, int, double *x, int *);
  int  SetPSiteClass(int iclass, double x[]);
#ifdef __cplusplus
}
#endif
#endif
