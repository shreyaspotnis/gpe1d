// CUDA implementation of the gpe1d code
// how to compile this coming up soon, I have something very specific on my
// computer
/*
Copyright 2012 Shreyas Potnis

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project

#include <cufft.h>
#include <cutil_inline.h>
#include <shrQATest.h>
#include "pca_utils.h"

#define BLOCKSIZE 512

typedef float2 Complex; 

__device__ float psi_sum_d;

// The same as the x unitary we had in our CPU version
static __global__ void x_unitary(int Nx, Complex *psiX, Complex *U1c, 
                                float C1)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads)
    {
        float an, es, ec, temp;
        an = C1*(psiX[i].x*psiX[i].x + psiX[i].y*psiX[i].y);

        es = sin(an);
        ec = cos(an);

        temp = psiX[i].x;
        psiX[i].x = ( ec * psiX[i].x - es * psiX[i].y );
        psiX[i].y = ( ec * psiX[i].y + es * temp );

        temp = psiX[i].x;
        psiX[i].x = ( U1c[i].x * psiX[i].x - U1c[i].y * psiX[i].y );
        psiX[i].y = ( U1c[i].x * psiX[i].y + U1c[i].y * temp );

    }

}

// The same as the x unitary we had in our CPU version
static __global__ void x_unitary_imag(int Nx, Complex *psiX, Complex *U1c, 
                                float C1)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads)
    {
        float an, ex, temp;
        an = C1*(psiX[i].x*psiX[i].x + psiX[i].y*psiX[i].y);

        ex = exp(an);

        psiX[i].x *= ex;
        psiX[i].y *= ex;

        temp = psiX[i].x;
        psiX[i].x = ( U1c[i].x * psiX[i].x - U1c[i].y * psiX[i].y );
        psiX[i].y = ( U1c[i].x * psiX[i].y + U1c[i].y * temp );

    }

}

static __global__ void k_unitary(int Nx, Complex *psiX, Complex *Kinc)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads)
    {
        float temp;
        temp = psiX[i].x;
        psiX[i].x = ( Kinc[i].x * psiX[i].x - Kinc[i].y * psiX[i].y );
        psiX[i].y = ( Kinc[i].x * psiX[i].y + Kinc[i].y * temp );

    }

}

static __global__ void psi_length(int Nx, Complex *psiX, float *sum_total, 
                float dx)
{
  __shared__ float  sum[BLOCKSIZE];
 
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  // Not needed, because NMAX is a power of two:
  //  if (i >= NMAX)
  //    return;
 
  sum[threadIdx.x] = psiX[i].x*psiX[i].x + psiX[i].y*psiX[i].y;

  // To make sure all threads in a block have the sum[] value:
  __syncthreads();

  int nTotalThreads = blockDim.x;  // Total number of active threads;
  // only the first half of the threads will be active.
  
  while(nTotalThreads > 1)
    {
      int halfPoint = (nTotalThreads >> 1);	// divide by two
 
      if (threadIdx.x < halfPoint)
	{
	  int thread2 = threadIdx.x + halfPoint;
	  sum[threadIdx.x] += sum[thread2];  // Pairwise summation
	}
      __syncthreads();
      nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }

  if (threadIdx.x == 0)
    {
      atomicAdd (sum_total, sum[0]*dx);
    }

  return;
}

static __global__ void normalize_psi(int Nx, Complex *psiX, float *sum_total)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads)
    {
        psiX[i].x /= sqrt(*sum_total);
        psiX[i].y /= sqrt(*sum_total);
    }
}

int main()
{
    int Nx, Ntstore, Ntskip;
    int memSize;
    int blockSize, nBlocks;
    float C1;
    float dx; 
    int imag_time;
    Complex *psiX, *U1c, *Kinc;
    Complex *psiX_d, *U1c_d, *Kinc_d;
    float *psi_sum_d;

    scanf("%d", &Nx);
    scanf("%d", &Ntstore);
    scanf("%d", &Ntskip);

    // allocate memory
    memSize = sizeof(Complex) * Nx;

    psiX = (Complex*)malloc(memSize);
    U1c = (Complex*)malloc(memSize);
    Kinc = (Complex*)malloc(memSize);

    // allocate memory on the device
    cutilSafeCall(cudaMalloc((void**)&psiX_d, memSize));
    cutilSafeCall(cudaMalloc((void**)&U1c_d, memSize));
    cutilSafeCall(cudaMalloc((void**)&Kinc_d, memSize));
    cutilSafeCall(cudaMalloc((void**)&psi_sum_d, sizeof(float)));


    for(int i=0; i<Nx; i++)
        scanf("%f %f", &psiX[i].x, &psiX[i].y);
    for(int i=0; i<Nx; i++)
        scanf("%f %f", &U1c[i].x, &U1c[i].y);
    for(int i=0; i<Nx; i++)
        scanf("%f %f", &Kinc[i].x, &Kinc[i].y);
    
    scanf("%f", &C1);
    scanf("%f", &dx);
    scanf("%d", &imag_time);
    
    // copy data to the device
    cutilSafeCall(cudaMemcpy(psiX_d, psiX , memSize,
                              cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(U1c_d, U1c, memSize,
                              cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(Kinc_d, Kinc , memSize,
                              cudaMemcpyHostToDevice));

    // CUFFT plan
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, Nx, CUFFT_C2C, 1));

    /* set up device execution configuration */
    blockSize = BLOCKSIZE;
    nBlocks = Nx / blockSize + (Nx % blockSize > 0);
    
    pca_time tt;
    tick(&tt);
    if(imag_time)
    {
        for(int t1=0; t1<Ntstore-1; t1++)
        {
            // run the core loop of the simulation
            for(int t2=0; t2<Ntskip; t2++)
            {

                x_unitary_imag<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);

                cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)psiX_d,
                                (cufftComplex *)psiX_d, CUFFT_FORWARD));
                k_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, Kinc_d);

                cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)psiX_d,
                                (cufftComplex *)psiX_d, CUFFT_INVERSE));

                x_unitary_imag<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);

                // set length of psi_sum_d to zero
                float zero_float = 0.0;
                cutilSafeCall(cudaMemcpy(psi_sum_d, &zero_float , sizeof(float),
                              cudaMemcpyHostToDevice));
               // normalize psiX_d
                psi_length<<<nBlocks, blockSize>>>(Nx, psiX_d, psi_sum_d, dx);
                normalize_psi<<<nBlocks, blockSize>>>(Nx, psiX_d, psi_sum_d);
        
               // cutilSafeCall(cudaMemcpy(&zero_float, psi_sum_d, sizeof(float),
               //               cudaMemcpyDeviceToHost));
               // fprintf(stderr, "%f %f \n", zero_float, dx);
           }
        
        }
        // send the output to stdout, our main process will catch it
        // get the data back from CUDA
        cutilSafeCall(cudaMemcpy(psiX, psiX_d, memSize,
                              cudaMemcpyDeviceToHost));

        printf("np.array([");
        for(int i=0; i<Nx; i++)
        {
            printf("%.10f + %.10fj, ", psiX[i].x, psiX[i].y);
        }

        printf("], complex)\n");
    }

    else
    {
        for(int t1=0; t1<Ntstore-1; t1++)
        {
            // run the core loop of the simulation
            for(int t2=0; t2<Ntskip; t2++)
            {

                x_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);

                cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)psiX_d,
                                (cufftComplex *)psiX_d, CUFFT_FORWARD));
                k_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, Kinc_d);

                cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)psiX_d,
                                (cufftComplex *)psiX_d, CUFFT_INVERSE));

                x_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);

                cudaDeviceSynchronize ();
            }
        
            // send the output to stdout, our main process will catch it
            // get the data back from CUDA
            cutilSafeCall(cudaMemcpy(psiX, psiX_d, memSize,
                                  cudaMemcpyDeviceToHost));

            printf("np.array([");
            for(int i=0; i<Nx; i++)
            {
                printf("%.10f + %.10fj, ", psiX[i].x, psiX[i].y);
            }

            printf("], complex)\n");
        }
    }
    tock(&tt);

    // release memory 
    cudaFree(psiX_d);
    cudaFree(U1c_d);
    cudaFree(Kinc_d);
    cudaFree(psi_sum_d);
    free(psiX);
    free(U1c);
    free(Kinc);

    return 0;
}
