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
    // FILLME: the naive GPU kernel code
}

/*
 *  The tiled GPU kernel(s) using global memory
 */ 
__global__ void GPU_KERNEL_NAME(_tiled_stage_X)(weight_t *dist, int n,
                                                int k_tile)
{
    // FILLME: tiled GPU kernel code for stage X
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

    /* FILLME: Set up and launch the kernel(s) */
    
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

graph_t *MAKE_KERNEL_NAME(_gpu, _tiled)(graph_t *graph)
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
