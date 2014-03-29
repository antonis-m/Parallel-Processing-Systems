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
__global__ void GPU_KERNEL_NAME(_tiled_stage_X)(weight_t *dist, int n,
        int k_tile, int kk)
{
    int tid = threadIdx.x*blockDim.y + threadIdx.y;  
    
    weight_t * a;
    weight_t * b;
    weight_t * c;
    
    /*case single block or column grid*/
    if (gridDim.x == 1) {

        /*signle block*/
        if (gridDim.y == 1) {
            a = &dist[k_tile*GPU_TILE_DIM+k_tile*GPU_TILE_DIM*n]; // Tkk
            b = a;
            c = a;
        } else { /*column grid*/
            if (blockIdx.y == k_tile)
                return;
            a = &dist[k_tile*GPU_TILE_DIM+blockIdx.y*GPU_TILE_DIM*n]; // Tik
            b = a;
            c = &dist[k_tile*GPU_TILE_DIM+k_tile*GPU_TILE_DIM*n]; //Tkk
        }

    } else { /*case column grid or square grid*/
        
        if (gridDim.y == 1) { /*line grid*/
            if (blockIdx.x == k_tile)
                return;
            a = &dist[k_tile*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM]; //Tki
            b = &dist[k_tile*GPU_TILE_DIM+k_tile*GPU_TILE_DIM*n]; //Tkk
            c = a;
        } else { /*square grid*/
            if ((blockIdx.x == k_tile) || (blockIdx.y == k_tile))
                return;
            a = &dist[blockIdx.y*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM]; //Tij
            b = &dist[blockIdx.y*n*GPU_TILE_DIM+k_tile*GPU_TILE_DIM]; //Tik
            c = &dist[k_tile*GPU_TILE_DIM*n+blockIdx.x*GPU_TILE_DIM]; //Tkj
        }

    }
    
    int row = tid / GPU_TILE_DIM;
    int col = tid % GPU_TILE_DIM;

    a[tid] = MIN(a[tid], b[row*n+kk]+c[kk*n+col]);  
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
__global__ void GPU_KERNEL_NAME(_tiled_shmem_stage_X)(weight_t *dist, int n,
                                                      int k_tile)
{
    // FILLME: tiled GPU kernel code using shared memory for stage X
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

    for (int kk=0;kk< tile_no;kk++) { // kk = K from the paper, be careful with sizes
    
        //phase one
        dim3 block(GPU_TILE_DIM, GPU_TILE_DIM);
        dim3 grid1(1);
        for(int k=0;k<GPU_TILE_DIM;k++) {
            GPU_KERNEL_NAME(_tiled_shmem)<<<grid1, block>>>(dist_gpu,graph->nr_vertices,k,kk);
            cudaThreadSynchronize();
        }

        dim3 grid2(tile_no);
        dim3 grid3(1,tile_no);
        for(int k=0;k<GPU_TILE_DIM;k++) {
            GPU_KERNEL_NAME(_tiled_shmem)<<<grid2, block>>>(dist_gpu,graph->nr_vertices,k,kk);
            cudaThreadSynchronize();
            GPU_KERNEL_NAME(_tiled_shmem)<<<grid3, block>>>(dist_gpu,graph->nr_vertices,k,kk);
            cudaThreadSynchronize();
        }

        dim3 grid4(tile_no,tile_no);
        for(int k=0;k<GPU_TILE_DIM;k++) {
            GPU_KERNEL_NAME(_tiled_shmem)<<<grid4, block>>>(dist_gpu,graph->nr_vertices,k,kk);
            cudaThreadSynchronize();
        }

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
