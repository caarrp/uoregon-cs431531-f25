#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"

template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz, 
                double* x, double* b)
{

    // COMPLETE THIS FUNCTION
   unsigned int row = blockIdx.x;
   unsigned int dim = blockDim.x; 
   unsigned int tid = threadIdx.x;
   unsigned int global_r = row*dim + tid;
    
   extern __shared__ double temp[];
   
   if (row < (unsigned int)m){//checking if row is in bounds
	double sum = 0.0;
    	//b[row] = 0.0;
	//accuumulator = 0

	unsigned int row_start = row * n;
	unsigned int row_end = row + n;

	for (unsigned int k = row_start + tid; k < row_end; k+= dim) {
            //int index = row + k * m;

            unsigned int col = col_ind[k];

            if (col != (unsigned int)n){
                sum += vals[k] * x[col];
            }
        }
	temp[tid] = sum;
	__syncthreads();
	//b[row] = 0.0;

	for (unsigned int j = dim/2; j > 0; j >>= 1){
		//reduction for the threads that did ta opertaions
	    if (tid > j){
	        unsigned int s = tid + j;
		temp[tid] += temp[s];}

	    __syncthreads();

	    if (tid == 0){
		    //only the one taht has the final sum, 
		b[row] = temp[0];}
    }
}}



void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    unsigned int blocks = m; 
    unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, nnz, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy ELL data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION
    unsigned int rows_used = 0;
    for (int i = 0; i < m * n; i++) {
        if (col_ind[i] != (unsigned int)-1 && col_ind[i] < (unsigned int)n) {
            rows_used++;
        }
    }

    unsigned int ell_array = rows_used * n;
    
    CopyData<unsigned int>(col_ind, ell_array, sizeof(unsigned int), dev_col_ind);
    CopyData<double>(vals, ell_array, sizeof(double), dev_vals);
    CopyData<double>(x, m, sizeof(double), dev_x);   
    
    checkCudaErrors(cudaMalloc((void**)dev_b, m * sizeof(double)));
    checkCudaErrors(cudaMemset(*dev_b, 0, m * sizeof(double))); 

    assert(*dev_col_ind != nullptr);
    assert(*dev_vals != nullptr);
    assert(*dev_x != nullptr);
    assert(*dev_b != nullptr);
}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy CSR data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION

    CopyData<unsigned int>(row_ptr, m + 1, sizeof(unsigned int), dev_row_ptr);
    CopyData<unsigned int>(col_ind, nnz, sizeof(unsigned int), dev_col_ind);
    CopyData<double>(vals, nnz, sizeof(double), dev_vals);
    CopyData<double>(x, n, sizeof(double), dev_x);
    
    checkCudaErrors(cudaMalloc((void**)dev_b, m * sizeof(double)));
    checkCudaErrors(cudaMemset(*dev_b, 0, m * sizeof(double)));

    assert(*dev_row_ptr != nullptr);
    assert(*dev_col_ind != nullptr);
    assert(*dev_vals != nullptr);
    assert(*dev_x != nullptr);
    assert(*dev_b != nullptr);
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
            int m, int n, int nnz, double* x, double* b)
{
    // COMPLETE THIS FUNCTION
    unsigned int row = blockIdx.x;
    unsigned int dim = blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int global_r = row*dim + tid;

    extern __shared__ double temp[];


    if (row < (unsigned int)m) {
        double sum = 0.0;
	
    	//b[row] = 0.0;
        unsigned int row_start = row_ptr[row];
        unsigned int row_end = row_ptr[row + 1];

        for (unsigned int j = row_start + tid; j < row_end; j+= dim) {

		unsigned int col_index = col_ind[j];
		
		if (col_index < (unsigned int)n) {
            	sum += vals[j] * x[col_index];
		}
	}

	temp[tid] = sum;
	
	//reducigng here
	for (unsigned int k = (dim / 2) ; k > 0; k>>=1){
	
	    if (tid < k) {
	        unsigned int s = tid + k;
                temp[tid] += temp[s];
		}
	    __syncthreads();
	}
	if (tid == 0){
	    b[row] = temp[0];
	}
    }
}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 thread block per row
    // 64 threads working on the non-zeros on the same row
    unsigned int blocks = m; 
    unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}
