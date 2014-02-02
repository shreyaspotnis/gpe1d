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
#include "pca_utils.h"

#define BLOCKSIZE 512 

// Set this during compile
#ifdef _USE_DOUBLE_PRECISION
typedef double2 Complex;
typedef double Ipp;
#else
typedef float2 Complex;
typedef float Ipp;
#endif

int readInt ( FILE *fp) {
    int a;
    fread (&a, 1, sizeof(int), fp );
    return a;
}

Ipp readFloat ( FILE *fp) {
    Ipp a;
    fprintf(stderr, "%d", sizeof(Ipp));
    fread (&a, 1, sizeof(Ipp), fp );
    return a;
}

struct InputData {
    int Nx;
    int Ntstore;
    int Ntskip;
    int imag_time;
    Ipp C1;
    Ipp dx;
};

// The same as the x unitary we had in our CPU version
static __global__ void x_unitary(int Nx, Complex *psiX, Complex *U1c, 
                                Ipp C1)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads) {
        Ipp an, es, ec, temp;
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
                                        Ipp C1)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads) {
        Ipp an, ex, temp;
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
    for (int i = threadID; i < Nx; i += numThreads) {
        Ipp temp;
        temp = psiX[i].x;
        psiX[i].x = ( Kinc[i].x * psiX[i].x - Kinc[i].y * psiX[i].y );
        psiX[i].y = ( Kinc[i].x * psiX[i].y + Kinc[i].y * temp );
    }
}
/*
static __global__ void psi_length(int Nx, Complex *psiX, Ipp *sum_total, 
                Ipp dx)
{
    // note: works only for Nx which are powers of 2
    __shared__ Ipp  sum[BLOCKSIZE];
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    sum[threadIdx.x] = psiX[i].x*psiX[i].x + psiX[i].y*psiX[i].y;
    // To make sure all threads in a block have the sum[] value:
    __syncthreads();
    int nTotalThreads = blockDim.x;  // Total number of active threads;
    // only the first half of the threads will be active.
    while(nTotalThreads > 1) {
        int halfPoint = (nTotalThreads >> 1);	// divide by two
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint;
            sum[threadIdx.x] += sum[thread2];  // Pairwise summation
        }
        __syncthreads();
        nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }

    if (threadIdx.x == 0) {
      atomicAdd (sum_total, sum[0]*dx);
    }

    return;
}*/

static __global__ void psi_block_length(int Nx, Complex* psiX, Ipp
                                        *psi_block_sum, Ipp dx) {
    // note: works only for Nx which are powers of 2
    __shared__ Ipp  sum[BLOCKSIZE];
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    sum[threadIdx.x] = psiX[i].x*psiX[i].x + psiX[i].y*psiX[i].y;
    // To make sure all threads in a block have the sum[] value:
    __syncthreads();
    int nTotalThreads = blockDim.x;  // Total number of active threads;
    // only the first half of the threads will be active.
    while(nTotalThreads > 1) {
        int halfPoint = (nTotalThreads >> 1);	// divide by two
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint;
            sum[threadIdx.x] += sum[thread2];  // Pairwise summation
        }
        __syncthreads();
        nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }
    if (threadIdx.x == 0)
        psi_block_sum[blockIdx.x] = sum[0] * dx;
    return;
}

static __global__ void psi_total_length(Ipp *psi_block_sum,
                                        Ipp *psi_total_sum) {
    extern __shared__ Ipp sum[];
    // Copying from global to shared memory: 
    sum[threadIdx.x] = psi_block_sum[threadIdx.x];
    // To make sure all threads in a block have the sum[] value:
    __syncthreads();
    int nTotalThreads = blockDim.x;  // Total number of active threads;
    // only the first half of the threads will be active.
    while(nTotalThreads > 1) {
        int halfPoint = (nTotalThreads >> 1); // divide by two
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint;
            sum[threadIdx.x] += sum[thread2];  // Pairwise summation
        }
        __syncthreads();
        nTotalThreads = halfPoint;
    }
    if (threadIdx.x == 0) {
        *psi_total_sum = sum[0];
    }
    return;
}

static __global__ void normalize_psi(int Nx, Complex *psiX, Ipp *sum_total)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads) {
        psiX[i].x /= sqrt(*sum_total);
        psiX[i].y /= sqrt(*sum_total);
    }
}

int main()
{
    fprintf(stderr, "Start program\n");
    InputData input;
    fread (&input, 1, sizeof(InputData), stdin);
    int Nx = input.Nx;
    int Ntstore = input.Ntstore;
    int Ntskip = input.Ntskip;
    int imag_time = input.imag_time;
    Ipp C1 = input.C1;
    Ipp dx = input.dx;

    /*
    int Nx = readInt(stdin);
    int Ntstore = readInt(stdin);
    int Ntskip = readInt(stdin);
    Ipp C1 = readFloat(stdin);
    Ipp dx = readFloat(stdin);
    int imag_time = readInt(stdin);*/
    fprintf(stderr, "have ints and doubles\n");
    fprintf(stderr, "Nx:%d Ntstore %d Ntskip %d C1 %f dx %f imag_time %d",
                    Nx, Ntstore, Ntskip, C1, dx, imag_time);
    
    int memSize;
    int blockSize, nBlocks;
    Complex *psiX, *U1c, *Kinc;
    Complex *psiX_d, *U1c_d, *Kinc_d;
    Ipp *psi_sum_d;
    Ipp *psi_block_sum_d;

    // allocate memory
    memSize = sizeof(Complex) * Nx;

    psiX = (Complex*)malloc(memSize);
    U1c = (Complex*)malloc(memSize);
    Kinc = (Complex*)malloc(memSize);

    // allocate memory on the device
    cudaMalloc((void**)&psiX_d, memSize);
    cudaMalloc((void**)&U1c_d, memSize);
    cudaMalloc((void**)&Kinc_d, memSize);
    cudaMalloc((void**)&psi_sum_d, sizeof(Ipp));

    fprintf(stderr, "size of psiX cuda:%d\n", Nx * sizeof(Complex));
    fread(psiX, Nx, sizeof(Complex), stdin);
    fprintf(stderr, "have psiX");

    fread(U1c, Nx, sizeof(Complex), stdin);
    fprintf(stderr, "have U1c");
    fread(Kinc, Nx, sizeof(Complex), stdin);
    fprintf(stderr, "have Kinc");

    fprintf(stderr, "have arrays\n");
    // copy data to the device
    cudaMemcpy(psiX_d, psiX , memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(U1c_d, U1c, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Kinc_d, Kinc , memSize, cudaMemcpyHostToDevice);

    // CUFFT plan
    cufftHandle plan;
#ifdef _USE_DOUBLE_PRECISION
    cufftPlan1d(&plan, Nx, CUFFT_Z2Z, 1);
#else
    cufftPlan1d(&plan, Nx, CUFFT_C2C, 1);
#endif

    /* set up device execution configuration */
    blockSize = BLOCKSIZE;
    nBlocks = Nx / blockSize + (Nx % blockSize > 0);
    cudaMalloc((void**)&psi_block_sum_d, nBlocks * sizeof(Ipp));
    // initialize block sum to zero
    cudaMemset(psi_block_sum_d, 0, nBlocks * sizeof(Ipp));
    
    pca_time tt;
    tick(&tt);
    if(!imag_time) {
            fwrite(psiX, Nx, sizeof(Complex), stdout);
    }
    for(int t1=0; t1<Ntstore-1; t1++) {
        for(int t2=0; t2<Ntskip; t2++) {
            if(imag_time)
                x_unitary_imag<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);
            else
                x_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);
            cufftExecC2C(plan, (cufftComplex *)psiX_d,
                            (cufftComplex *)psiX_d, CUFFT_FORWARD);
            k_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, Kinc_d);
            cufftExecC2C(plan, (cufftComplex *)psiX_d,
                            (cufftComplex *)psiX_d, CUFFT_INVERSE);
            if(imag_time)
                x_unitary_imag<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);
            else
                x_unitary<<<nBlocks, blockSize>>>(Nx, psiX_d, U1c_d, C1);
            cudaDeviceSynchronize ();

            if(imag_time) {
                // psi_length<<<nBlocks, blockSize>>>(Nx, psiX_d, psi_sum_d, dx);
                psi_block_length<<<nBlocks, blockSize>>>(Nx, psiX_d,
                                                         psi_block_sum_d, dx); 
                psi_total_length<<<1, nBlocks, nBlocks * sizeof(Ipp)>>>
                                                (psi_block_sum_d, psi_sum_d);

                normalize_psi<<<nBlocks, blockSize>>>(Nx, psiX_d, psi_sum_d);
             }               
        }
        if(!imag_time) {
            // send the output to stdout, our main process will catch it
            cudaMemcpy(psiX, psiX_d, memSize, cudaMemcpyDeviceToHost);
            fwrite(psiX, Nx, sizeof(Complex), stdout);
        }
    }
    if(imag_time) {
        cudaMemcpy(psiX, psiX_d, memSize, cudaMemcpyDeviceToHost);
        fwrite(psiX, Nx, sizeof(Complex), stdout);
    }

    tock(&tt);

    // release memory 
    cudaFree(psiX_d);
    cudaFree(U1c_d);
    cudaFree(Kinc_d);
    cudaFree(psi_sum_d);
    cudaFree(psi_block_sum_d);
    free(psiX);
    free(U1c);
    free(Kinc);
    return 0;
}
