#include <math.h>
#include <immintrin.h>
#include "codeml.h"

int OMP_NodeScale(int inode, int pos0, int pos1)
{
    /* scale to avoid underflow */
   int n = com.ncode;
   int k = nodeScaleK[inode];
   int h;
#pragma omp parallel for private(h)
   for(h=pos0; h<pos1; h++) {
       double t = 0;
       int j;
       for(j=0;j<n;j++)
           if(nodes_conP[inode][h*64+j]>t)
               t = nodes_conP[inode][h*64+j];

       if(t<1e-300) {
           for(j=0;j<n;j++)
               nodes_conP[inode][h*64+j]=1;  /* both 0 and 1 fine */
           com.nodeScaleF[k*com.npatt+h] = -800;  /* this is problematic? */
       }
       else {  
           for(j=0;j<n;j++)
               nodes_conP[inode][h*64+j]/=t;
           com.nodeScaleF[k*com.npatt+h] = log(t);
       }
   }

   return 0;
}

int OMP_ConditionalPNode(int n, int pos0, int pos1)
{
	/*
      Calculate the contidional likelihood of each internal node in post order. 
	*/
    int tail;
    for (tail = tree.nnode - 1; tail >= 0; tail--) {
        int inode = bfs_que[tail];
        if (inode < com.ns) continue;

        int h;
#pragma omp parallel for private(h)
        for (h = pos0 * 64; h < pos1 * 64; h++) {
            nodes_conP[inode][h] = 1;
        }

        int i;
        for (i = 0; i < nodes[inode].nson; i++) {
            int ison = nodes[inode].sons[i];

            int h;
            if (nodes[ison].nson < 1 && com.cleandata) {        /* tip && clean */
#pragma omp parallel for private(h)
                for (h = pos0; h < pos1; h++) {
                    int j;
                    for (j = 0; j < n; j++) {
                        nodes_conP[inode][h * 64 + j] *= PMatAll[ison][com.z[ison][h] * 64 + j];
                    }
                }
            } else if (nodes[ison].nson < 1 && !com.cleandata) {  /* tip & unclean */
#pragma omp parallel for private(h)
                for (h = pos0; h < pos1; h++) {
                    int j, k;
                    for (j = 0; j < n; j++) {
                        double t = 0;
                        for (k = 0; k < nChara[com.z[ison][h]]; k++)
                            t += PMatAll[ison][CharaMap[com.z[ison][h]][k] * 64 + j];
                        nodes_conP[inode][h * 64 + j] *= t;
                    }
                }
            } else {                                            /* internal node */
#ifdef SSE
                __m128d *pPMat = (__m128d *) PMatAll[ison];
                __m128d *pConP = (__m128d *) nodes_conP[inode];
#pragma omp parallel for private(h)
                for (h = pos0; h < pos1; h++) {
                    double __attribute((aligned(0x10))) sum[64] = {0};
                    __m128d *pSum = (__m128d *) sum;
                    __m128d __attribute((aligned(0x10))) conP;

                    int j, k;
                    for (k = 0; k < n; k++) {
                        conP = _mm_set_pd(nodes_conP[ison][h * 64 + k], nodes_conP[ison][h * 64 + k]);
                        for (j = 0; j < 32; j++) {
                            pSum[j] = _mm_add_pd(pSum[j], _mm_mul_pd(pPMat[k * 32 + j], conP));
                        }
                    }

                    for (j = 0; j < 32; j++) {
                        pConP[h * 32 + j] = _mm_mul_pd(pConP[h * 32 + j], pSum[j]);
                    }
                }
#else
#pragma omp parallel for private(h)
                for (h = pos0; h < pos1; h++) {
                    int j, k;
                    for (j = 0; j < n; j++) {
                        double t = 0;
                        for(k = 0; k < n; k++)
                            t += PMatAll[ison][k * 64 + j] * nodes_conP[ison][h * 64 + k];
                        nodes_conP[inode][h * 64 + j] *= t;
                    }
                }
#endif
            }
        }        /*  for (ison)  */
        if (com.NnodeScale && com.nodeScale[inode]) {
            OMP_NodeScale(inode, pos0, pos1);
        }
    }

    return 0;
}

void OMP_CalcFhk(double *fhK, int pos0, int pos1)
{
	/*
	  Intergrate the conditional likelihood at the root.
	*/
    int n = com.ncode;
    int m = com.npatt;

    OMP_ConditionalPNode(n, pos0, pos1);

    int h;
#pragma omp parallel for private(h)
    for (h = pos0; h < pos1; h++) {
        double fh = 0;
        int i;
        for (i = 0; i < n; i++) {
            fh += com.pi[i] * nodes_conP[tree.root][h * 64 + i];
        }
        if (fh <= 0) {
            fh = 1e-300;
        }
        if (!com.NnodeScale) {
            fhK[h] = fh;
        } else {
            fhK[h] = log(fh);
            int k;
            for (k = 0; k < com.NnodeScale; k++) {
                fhK[h] += com.nodeScaleF[k * m + h];
            }
        }
    }
}
