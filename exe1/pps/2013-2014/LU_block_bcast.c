#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include "utils.h"


int main (int argc, char * argv[]) {
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int X,Y,x,y,X_ext,i,j,k;
    double ** A, ** localA;
    X=atoi(argv[1]);
    Y=X;
    if (rank==0) {
        //Allocate and init matrix A
        A=malloc2D(X,Y);
        init2D(A,X,Y);
        char * name="initial_block_bcast";
        print2DFile(A,X,Y,name);
    }

    //Extend dimension X with ghost cells if X%size!=0
    if (X%size!=0)
        X_ext=X+size-X%size;
    else
        X_ext=X;
      
    //Local dimensions x,y
    x=X_ext/size;
    y=Y;

    //Allocate local matrix and scatter global matrix
    localA=malloc2D(x,y);
    double * idx;
    if (rank==0)
        idx=&A[0][0];
    MPI_Scatter(idx,x*y,MPI_DOUBLE,&localA[0][0],x*y,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if (rank==0)
        free2D(A,X,Y);
 
    //Timers   
    struct timeval ts,tf,comps,compf,comms,commf;
    double total_time,computation_time,communication_time;

	MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&ts,NULL);        
    /******************************************************************************
     The matrix A is distributed in contiguous blocks to the local matrices localA
     You have to use collective communication routines.
     Don't forget the timers for computation and communication!
	
	 ******************************************************************************/
	double * temp = malloc(y*sizeof(double));
	double l;
	computation_time = 0;
	communication_time = 0;

	for (k = 0; k < X - 1; k++) {
		// find which rank must send and copy the correct row and size
		if (rank == (k / size)){
			printf("rank=%d\n", rank);
			 memcpy(&temp[k], &localA[k%size][k], (y-k)*sizeof(double));   // this is an optimization
			 //memcpy(temp, &localA[k%size][0], y*sizeof(double));
			 }
		//send
		gettimeofday(&comms, NULL);
		MPI_Bcast(&temp[k], y-k, MPI_DOUBLE, k/size, MPI_COMM_WORLD);
		gettimeofday(&commf, NULL);
		communication_time+=commf.tv_sec-comms.tv_sec+(commf.tv_usec-comms.tv_usec)*0.000001;

		//if done with rows stop
		if (k>=((rank+1)*size))
			goto OUT;

		gettimeofday(&comps, NULL);	
		if (rank == (k/size)) 
			for (i = k%size+1; i < x; i++){
				l = localA[i][k] / temp[k];
				for (j = k+1; j < y; j++) {
					printf("rank = %d i = %d j = %d\n", rank, i, j);
					localA[i][j] = localA[i][j] -l*temp[j];
				}
			}
		else 
			for (i = 0; i < x; i++){
				l = localA[i][k] / temp[k];
				for (j = k+1; j < y; j++) {
					printf("rank = %d i = %d j = %d\n", rank, i, j);
					localA[i][j] = localA[i][j] -l*temp[j];
				}
			}
		gettimeofday(&compf, NULL);	
		computation_time+=compf.tv_sec-comps.tv_sec+(compf.tv_usec-comps.tv_usec)*0.000001;
		OUT:
		MPI_Barrier(MPI_COMM_WORLD);
	}
    gettimeofday(&tf,NULL);
    total_time=tf.tv_sec-ts.tv_sec+(tf.tv_usec-ts.tv_usec)*0.000001;
	

    //Gather local matrices back to the global matrix
    if (rank==0) {
        A=malloc2D(X,Y);
        idx=&A[0][0];
    }
    MPI_Gather(&localA[0][0],x*y,MPI_DOUBLE,idx,x*y,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    double avg_total,avg_comp,avg_comm,max_total,max_comp,max_comm;
    MPI_Reduce(&total_time,&max_total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&computation_time,&max_comp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&total_time,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&computation_time,&avg_comp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&avg_comm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    avg_total/=size;
    avg_comp/=size;
    avg_comm/=size;

    if (rank==0) {
        printf("LU-Block-bcast\tSize\t%d\tProcesses\t%d\n",X,size);
        printf("Max times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",max_total,max_comp,max_comp);
        printf("Avg times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",avg_total,avg_comp,avg_comp);
    }

    //Print triangular matrix U to file
    if (rank==0) {
        char * filename="output_block_bcast";
        print2DFile(A,X,Y,filename);
    }
    
    MPI_Finalize();

    return 0;
}


