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

typedef float2 Complex; 
__device__ float psi_sum_d;

int readInt ( FILE *fp) {
    int a;
    fread (&a, 1, sizeof(int), fp );
    return a;
}

float readFloat ( FILE *fp) {
    float a;
    fread (&a, 1, sizeof(float), fp );
    return a;
}

// The same as the x unitary we had in our CPU version
static __global__ void x_unitary(int Nx, Complex *psiX, Complex *U1c, 
                                float C1)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < Nx; i += numThreads) {
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
    for (int i = threadID; i < Nx; i += numThreads) {
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
    for (int i = threadID; i < Nx; i += numThreads) {
        float temp;
        temp = psiX[i].x;
        psiX[i].x = ( Kinc[i].x * psiX[i].x - Kinc[i].y * psiX[i].y );
        psiX[i].y = ( Kinc[i].x * psiX[i].y + Kinc[i].y * temp );
    }
}

static __global__ void psi_length(int Nx, Complex *psiX, float *sum_total, 
                float dx)
{
    // note: works only for Nx which are powers of 2
    __shared__ float  sum[BLOCKSIZE];
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
}

static __global__ void normalize_psi(int Nx, Complex *psiX, float *sum_total)
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
    int Nx = readInt(stdin);
    int Ntstore = readInt(stdin);
    int Ntskip = readInt(stdin);
    float C1 = readFloat(stdin);
    float dx = readFloat(stdin);
    int imag_time = readInt(stdin);
    
    int memSize;
    int blockSize, nBlocks;
    Complex *psiX, *U1c, *Kinc;
    Complex *psiX_d, *U1c_d, *Kinc_d;
    float *psi_sum_d;

    // allocate memory
    memSize = sizeof(Complex) * Nx;

    psiX = (Complex*)malloc(memSize);
    U1c = (Complex*)malloc(memSize);
    Kinc = (Complex*)malloc(memSize);

    // allocate memory on the device
    cudaMalloc((void**)&psiX_d, memSize);
    cudaMalloc((void**)&U1c_d, memSize);
    cudaMalloc((void**)&Kinc_d, memSize);
    cudaMalloc((void**)&psi_sum_d, sizeof(float));

    fread(psiX, Nx, sizeof(Complex), stdin);
    fread(U1c, Nx, sizeof(Complex), stdin);
    fread(Kinc, Nx, sizeof(Complex), stdin);

    // copy data to the device
    cudaMemcpy(psiX_d, psiX , memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(U1c_d, U1c, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Kinc_d, Kinc , memSize, cudaMemcpyHostToDevice);

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, Nx, CUFFT_C2C, 1);

    /* set up device execution configuration */
    blockSize = BLOCKSIZE;
    nBlocks = Nx / blockSize + (Nx % blockSize > 0);
    
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
                float zero_float = 0.0;
                cudaMemcpy(psi_sum_d, &zero_float , sizeof(float),
                              cudaMemcpyHostToDevice);
                psi_length<<<nBlocks, blockSize>>>(Nx, psiX_d, psi_sum_d, dx);
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
    free(psiX);
    free(U1c);
    free(Kinc);
    return 0;
}
