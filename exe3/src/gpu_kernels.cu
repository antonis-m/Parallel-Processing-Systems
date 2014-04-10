/*
 *  gpu_kernels.cu -- GPU kernels
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Vasileios Karakasis
 */ 

#include <stdio.h>
#include <cuda.h>
#include "error.h"
#include "gpu_util.h"
#include "graph.h"
#include "timer.h"

#define GPU_KERNEL_NAME(name)   do_apsp_gpu ## name

weight_t *copy_graph_to_gpu(const graph_t *graph)
{
    size_t dist_size = graph->nr_vertices*graph->nr_vertices;
    weight_t *dist_gpu = (weight_t *) gpu_alloc(dist_size*sizeof(*dist_gpu));
    if (!dist_gpu)
        error(0, "gpu_alloc() failed: %s", gpu_get_last_errmsg());

    if (copy_to_gpu(graph->weights[0], dist_gpu,
                    dist_size*sizeof(*dist_gpu)) < 0)
        error(0, "copy_to_gpu() failed: %s", gpu_get_last_errmsg());

    return dist_gpu;
}

graph_t *copy_graph_from_gpu(const weight_t *dist_gpu, graph_t *graph)
{
    size_t dist_size = graph->nr_vertices*graph->nr_vertices;

    if (copy_from_gpu(graph->weights[0], dist_gpu,
                      dist_size*sizeof(*dist_gpu)) < 0)
        error(0, "copy_from_gpu() failed: %s", gpu_get_last_errmsg());

    return graph;
}

/*
 *  The naive GPU kernel
 */ 
__global__ void GPU_KERNEL_NAME(_naive)(weight_t *dist, int n, int k)
{
    int tid = (blockDim.x*blockDim.y*blockIdx.x)+ // consider line grid
        (threadIdx.x*blockDim.y + threadIdx.y);   // and global order 

    if (tid > n*n)
        return;

    int row = tid / n;
    int col = tid % n;

    dist[tid] = MIN(dist[tid], dist[row*n+k]+dist[k*n+col]);  
}

/*
 *  The tiled GPU kernel(s) using global memory
 */ 
__global__ void GPU_KERNEL_NAME(_tiled_stage_1)(weight_t *dist, int n,
        int k_tile)
{
    weight_t * a;
    weight_t * b;
    weight_t * c;
    /*single block*/
    a = &dist[k_tile*GPU_TILE_DIM+k_tile*GPU_TILE_DIM*n]; // Tkk
    b = a;
    c = a;

    int row = threadIdx.x;  // row-cal in the small square
    int col = threadIdx.y;
    
    for (int kk =0;kk<GPU_TILE_DIM;kk++) { 
        a[row*n+col] = MIN(a[row*n+col], b[row*n+kk]+c[kk*n+col]);
        __syncthreads();
    }
}


__global__ void GPU_KERNEL_NAME(_tiled_stage_2)(weight_t *dist, int n,
        int k_tile)
{

    weight_t * a;
    weight_t * b;
    weight_t * c;
    /*column grid*/
    if (blockIdx.y == k_tile)
        return;
    a = &dist[k_tile*GPU_TILE_DIM+blockIdx.y*GPU_TILE_DIM*n]; // Tik
    b = a;
    c = &dist[k_tile*GPU_TILE_DIM+k_tile*GPU_TILE_DIM*n]; //Tkk

    int row = threadIdx.x;  // row-cal in the small square
    int col = threadIdx.y;
    
    for (int kk=0;kk<GPU_TILE_DIM;kk++) {
        a[row*n+col] = MIN(a[row*n+col], b[row*n+kk]+c[kk*n+col]);
        __syncthreads();
    }
}

__global__ void GPU_KERNEL_NAME(_tiled_stage_3)(weight_t *dist, int n,
        int k_tile)
{
    weight_t * a;
    weight_t * b;
    weight_t * c;
    /*line grid*/
    if (blockIdx.x == k_tile)
        return;
    a = &dist[k_tile*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM]; //Tki
    b = &dist[k_tile*GPU_TILE_DIM+k_tile*GPU_TILE_DIM*n]; //Tkk
    c = a;

    int row = threadIdx.x;  // row-cal in the small square
    int col = threadIdx.y;

    for (int kk=0;kk<GPU_TILE_DIM;kk++) {
        a[row*n+col] = MIN(a[row*n+col], b[row*n+kk]+c[kk*n+col]);
        __syncthreads();
    }
}
__global__ void GPU_KERNEL_NAME(_tiled_stage_4)(weight_t *dist, int n,
        int k_tile)
{
    weight_t * a;
    weight_t * b;
    weight_t * c;
    /*square grid*/
    if ((blockIdx.x == k_tile) || (blockIdx.y == k_tile))
        return;
    a = &dist[blockIdx.y*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM]; //Tij
    b = &dist[blockIdx.y*n*GPU_TILE_DIM+k_tile*GPU_TILE_DIM]; //Tik
    c = &dist[k_tile*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM]; //Tkj

    int row = threadIdx.x;  // row-cal in the small square
    int col = threadIdx.y;
    
    for (int kk=0;kk<GPU_TILE_DIM;kk++) {
        a[row*n+col] = MIN(a[row*n+col], b[row*n+kk]+c[kk*n+col]);
        __syncthreads();
    }
}

/*
 *  FILLME: Use different kernels for the different stages of the
 *  tiled FW computation
 *
 *  Use GPU_TILE_DIM (see graph.h) as the tile dimension. You can
 *  adjust its value during compilation. See `make help' for more
 *  information.
 */ 

/*
 *  The tiled GPU kernel(s) using shared memory
 */ 
__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_1)(weight_t *dist, int n,
                                                      int k_tile){

    int tid = threadIdx.x*blockDim.y + threadIdx.y;

    __shared__ weight_t  a [GPU_TILE_DIM * GPU_TILE_DIM];
    __shared__ weight_t  * b;
    __shared__ weight_t  * c;


    int row = tid / GPU_TILE_DIM;  // row-cal in the small square
    int col = tid % GPU_TILE_DIM;
    
    a[row*GPU_TILE_DIM + col] = dist[k_tile*GPU_TILE_DIM + k_tile*GPU_TILE_DIM*n + row*n + col];
    b=a;
    c=a; 
    __syncthreads();
 
    for (int kk =0;kk<GPU_TILE_DIM;kk++) { 
        a[row*GPU_TILE_DIM+col] = MIN(a[row*GPU_TILE_DIM+col], b[row*GPU_TILE_DIM+kk]+c[kk*GPU_TILE_DIM+col]);
        __syncthreads();
    }

    dist[k_tile*GPU_TILE_DIM + k_tile*GPU_TILE_DIM*n + row*n + col] = a[row*GPU_TILE_DIM + col] ;
   
}

__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_2)(weight_t *dist, int n, int k_tile){

    int tid = threadIdx.x*blockDim.y + threadIdx.y;

    __shared__ weight_t  a[GPU_TILE_DIM * GPU_TILE_DIM];
    __shared__ weight_t * b;
    __shared__ weight_t  c[GPU_TILE_DIM * GPU_TILE_DIM];

    /*column grid*/
    if (blockIdx.y == k_tile)
        return;

    int row = tid / GPU_TILE_DIM;  // row-cal in the small square
    int col = tid % GPU_TILE_DIM;

    a[row*GPU_TILE_DIM+col] = dist[k_tile*GPU_TILE_DIM + blockIdx.y*GPU_TILE_DIM*n + row*n + col];
    __syncthreads();
    b=a;
    c[row*GPU_TILE_DIM+col] = dist[k_tile*GPU_TILE_DIM + k_tile*GPU_TILE_DIM*n + row*n + col];
    __syncthreads();
    
    for (int kk=0;kk<GPU_TILE_DIM;kk++) {
        a[row*GPU_TILE_DIM+col] = MIN(a[row*GPU_TILE_DIM+col], b[row*GPU_TILE_DIM+kk]+c[kk*GPU_TILE_DIM+col]);
        __syncthreads();
    }
      
    dist[k_tile*GPU_TILE_DIM + blockIdx.y*GPU_TILE_DIM*n +row*n+col] = a[row*GPU_TILE_DIM + col] ;
    __syncthreads(); 
}

__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_3)(weight_t *dist, int n, int k_tile){

    int tid = threadIdx.x*blockDim.y + threadIdx.y;

    __shared__  weight_t  a[GPU_TILE_DIM*GPU_TILE_DIM] ;
    __shared__  weight_t  b[GPU_TILE_DIM*GPU_TILE_DIM];
    __shared__  weight_t * c;

    /*line grid*/
    if (blockIdx.x == k_tile)
        return;

    int row = tid / GPU_TILE_DIM;  // row-cal in the small square
    int col = tid % GPU_TILE_DIM;

    a[row*GPU_TILE_DIM +col] = dist[k_tile*GPU_TILE_DIM*n + blockIdx.x*GPU_TILE_DIM +row*n + col];
    __syncthreads();
    b[row*GPU_TILE_DIM +col]= dist[k_tile*GPU_TILE_DIM + k_tile*GPU_TILE_DIM*n + row*n +col];
    __syncthreads();
    c=a;
 
    for (int kk=0;kk<GPU_TILE_DIM;kk++) {
        a[row*GPU_TILE_DIM+col] = MIN(a[row*GPU_TILE_DIM+col], b[row*GPU_TILE_DIM+kk]+c[kk*GPU_TILE_DIM+col]);
        __syncthreads();
    }

    dist[k_tile*GPU_TILE_DIM*n + blockIdx.x*GPU_TILE_DIM +row*n + col] = a[row*GPU_TILE_DIM + col];
    __syncthreads(); 
}


__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_4)(weight_t *dist, int n, int k_tile){

    int tid = threadIdx.x*blockDim.y + threadIdx.y;

    __shared__  weight_t  a[GPU_TILE_DIM*GPU_TILE_DIM];
    __shared__  weight_t  b[GPU_TILE_DIM*GPU_TILE_DIM];
    __shared__  weight_t  c[GPU_TILE_DIM*GPU_TILE_DIM];

    /*square grid*/
    if ((blockIdx.x == k_tile) || (blockIdx.y == k_tile))
        return;
  
    int row = tid / GPU_TILE_DIM;  // row-cal in the small square
    int col = tid % GPU_TILE_DIM;

    a[row*GPU_TILE_DIM+col] = dist[blockIdx.y*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM + row*n + col]; //Tij
    b[row*GPU_TILE_DIM+col] = dist[blockIdx.y*n*GPU_TILE_DIM+k_tile*GPU_TILE_DIM + row*n + col]; //Tik
    c[row*GPU_TILE_DIM+col] = dist[k_tile*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM + row*n + col]; //Tkj
    __syncthreads();

    for (int kk=0;kk<GPU_TILE_DIM;kk++) {
        a[row*GPU_TILE_DIM+col] = MIN(a[row*GPU_TILE_DIM+col], b[row*GPU_TILE_DIM+kk]+c[kk*GPU_TILE_DIM+col]);
        __syncthreads();
    }

    dist[blockIdx.y*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM + row*n + col] = a[row*GPU_TILE_DIM + col]; //Tij
    __syncthreads();
}
/*
 *  FILLME: Use different kernels for the different stages of the
 *  tiled FW computation
 *  
 *  Use GPU_TILE_DIM (see graph.h) as the tile dimension. You can
 *  adjust its value during compilation. See `make help' for more
 *  information.
 */ 

graph_t *MAKE_KERNEL_NAME(_gpu, _naive)(graph_t *graph)
{
    xtimer_t transfer_timer;
    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    weight_t *dist_gpu = copy_graph_to_gpu(graph);
    timer_stop(&transfer_timer);
    
    //init block and grid
    dim3 block(8,8);
    dim3 grid((graph->nr_vertices*graph->nr_vertices)/64); // this should change

    //call the GPU kernel
    for(int k=0;k<graph->nr_vertices;k++) { //main loop
        GPU_KERNEL_NAME(_naive)<<<grid, block>>>(dist_gpu,graph->nr_vertices,k);
        cudaThreadSynchronize();
    }

     /* Wait for last kernel to finish, so as to measure correctly the
     * transfer times Otherwise, copy from GPU will block
     */
    cudaThreadSynchronize();

    /* Copy back results to host */
    timer_start(&transfer_timer);
    copy_graph_from_gpu(dist_gpu, graph);
    timer_stop(&transfer_timer);
    printf("Total transfer times: %lf s\n",
           timer_elapsed_time(&transfer_timer));
    return graph;
}

graph_t *MAKE_KERNEL_NAME(_gpu, _tiled)(graph_t *graph)
{
    xtimer_t transfer_timer;
    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    weight_t *dist_gpu = copy_graph_to_gpu(graph);
    timer_stop(&transfer_timer);

    int tile_no = graph->nr_vertices / GPU_TILE_DIM;
    for (int k=0;k< tile_no;k++) { // k = K from the paper, be careful with sizes

        //phase one
        dim3 block(GPU_TILE_DIM, GPU_TILE_DIM);
        dim3 grid1(1);
        GPU_KERNEL_NAME(_tiled_stage_1)<<<grid1, block>>>(dist_gpu,graph->nr_vertices,k);
             
        //phase two
        dim3 grid2(1,tile_no);
        GPU_KERNEL_NAME(_tiled_stage_2)<<<grid2, block>>>(dist_gpu,graph->nr_vertices,k);       
        
        //phase three    
        dim3 grid3(tile_no);
        GPU_KERNEL_NAME(_tiled_stage_3)<<<grid3, block>>>(dist_gpu,graph->nr_vertices,k);

        dim3 grid4(tile_no,tile_no);
        GPU_KERNEL_NAME(_tiled_stage_4)<<<grid4, block>>>(dist_gpu,graph->nr_vertices,k);
    }
    /*
     * Wait for last kernel to finish, so as to measure correctly the
     * transfer times Otherwise, copy from GPU will block
     */
    cudaThreadSynchronize();

    /* Copy back results to host */
    timer_start(&transfer_timer);
    copy_graph_from_gpu(dist_gpu, graph);
    timer_stop(&transfer_timer);
    printf("Total transfer times: %lf s\n",
            timer_elapsed_time(&transfer_timer));
    return graph;
}

graph_t *MAKE_KERNEL_NAME(_gpu, _tiled_shmem)(graph_t *graph)
{
    xtimer_t transfer_timer;
    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    weight_t *dist_gpu = copy_graph_to_gpu(graph);
    timer_stop(&transfer_timer);

    int tile_no = graph->nr_vertices / GPU_TILE_DIM;
    for (int k=0;k< tile_no;k++) { // k = K from the paper, be careful with sizes

        //phase one
        dim3 block(GPU_TILE_DIM, GPU_TILE_DIM);
        dim3 grid1(1);
        GPU_KERNEL_NAME(_tiled_shmem_stage_1)<<<grid1, block>>>(dist_gpu,graph->nr_vertices,k);        

        //phase two
        dim3 grid2(1,tile_no);
        GPU_KERNEL_NAME(_tiled_shmem_stage_2)<<<grid2, block>>>(dist_gpu,graph->nr_vertices,k);        
        
        //phase three    
        dim3 grid3(tile_no);
        GPU_KERNEL_NAME(_tiled_shmem_stage_3)<<<grid3, block>>>(dist_gpu,graph->nr_vertices,k);

        dim3 grid4(tile_no,tile_no);
        GPU_KERNEL_NAME(_tiled_shmem_stage_4)<<<grid4, block>>>(dist_gpu,graph->nr_vertices,k);
    }

    /*
     * FILLME: Set up and launch the kernel(s)
     *
     * You may need different grid/block configurations for each stage
     * of the computation
     * 
     * Use GPU_TILE_DIM (see graph.h) as the tile dimension. You can
     * adjust its value during compilation. See `make help' for more
     * information.
     */

    /*
     * Wait for last kernel to finish, so as to measure correctly the
     * transfer times Otherwise, copy from GPU will block
     */
    cudaThreadSynchronize();

    /* Copy back results to host */
    timer_start(&transfer_timer);
    copy_graph_from_gpu(dist_gpu, graph);
    timer_stop(&transfer_timer);
    printf("Total transfer times: %lf s\n",
           timer_elapsed_time(&transfer_timer));
    return graph;
            
}
