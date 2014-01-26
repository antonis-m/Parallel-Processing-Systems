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
    MPI_Status *stat;
    MPI_Request *requestList, requestNull;
    
    int X,Y,x,y,X_ext,i,j,k,l,block_size;
    double L;
    double ** A, ** localA;
    X=atoi(argv[1]);
    Y=X;
    //Extend dimension X with ghost cells if X%size!=0
    if (X%size!=0)
        X_ext=X+size-X%size;
    else
        X_ext=X;
      
    //Local dimensions x,y
    x=X_ext/size;
    y=Y;
    
    if (rank==0) {
        //Allocate and init matrix A
        A=malloc2D(X_ext,Y);
        init2D(A,X,Y);
    }
    
    //Allocate local matrix and scatter global matrix
    localA=malloc2D(x,y);
    double * idx;
    if (rank==0) 
        idx=&A[0][0];
    MPI_Scatter(idx,x*y,MPI_DOUBLE,&localA[0][0],x*y,MPI_DOUBLE,0,MPI_COMM_WORLD);
 
   if (rank==0) {
        free2D(A,X_ext,Y);
    }

    //Timers   
    struct timeval ts,tf,comps,compf,comms,commf;
    double total_time,computation_time,communication_time;

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&ts,NULL);        

    /******************************************************************************
     The matrix A is distributed in contiguous blocks to the local matrices localA
     You have to use point-to-point communication routines    
     Don't forget to set the timers for computation and communication!
    ******************************************************************************/

 
     block_size= x;
     requestList =(MPI_Request*)malloc(size*sizeof(MPI_Request)); 
     stat = (MPI_Status*)malloc(size*sizeof(MPI_Status));
     MPI_Status status;
         for (k=0; k<(X_ext-1); k++) {   
         if (rank==k/block_size) {
           for(l=(k/block_size); l<size; l++){  
             if(l!=(k/block_size)) { 
               
               gettimeofday(&comms, NULL);
              //MPI_Send(&localA[k%x][k],y-k,MPI_DOUBLE,l,0,MPI_COMM_WORLD);    // replace with non blocking; 
               MPI_Isend(&localA[k%x][k],y-k,MPI_DOUBLE,l,0,MPI_COMM_WORLD,&requestNull);
               gettimeofday(&commf, NULL);
               communication_time+=commf.tv_sec-comms.tv_sec+(commf.tv_usec-comms.tv_usec)*0.000001;

               }
           }

           //computations
           gettimeofday(&comps, NULL);
           for (i=(k+1)%x; i<x; i++) {
             if (i==0)
                break;
             else {  
             L=localA[i][k]/localA[k%x][k];
             for (j=k; j<y; j++)
               localA[i][j]-=L*localA[k%x][j];
            }
           }
        //end of comp
        gettimeofday(&compf, NULL);
        computation_time+=compf.tv_sec-comps.tv_sec+(compf.tv_usec-comps.tv_usec)*0.000001;


       }  

       else if (rank > (k/block_size)) {
         
         double * line_received ;
         line_received = (double *)malloc(y*sizeof(double));

         gettimeofday(&comms, NULL);
         MPI_Recv(&line_received[0],y-k,MPI_DOUBLE,k/block_size,0,MPI_COMM_WORLD,&status);  // replace with non blocking ;
         gettimeofday(&commf, NULL);
         communication_time+=commf.tv_sec-comms.tv_sec+(commf.tv_usec-comms.tv_usec)*0.000001;

         //MPI_Irecv(&line_received[0],y-k,MPI_DOUBLE,k/block_size,0,MPI_COMM_WORLD,&requestList[rank-1]);
         //MPI_Wait(&requestList[rank-1], &stat[rank-1]);     //added Waitall         

         //computations
         gettimeofday(&comps, NULL);
         for (i=0; i<x; i++) {
           L=localA[i][k]/line_received[0];
           for (j=k; j<y; j++)
             localA[i][j]-=L*line_received[j-k];
          }
        gettimeofday(&compf, NULL);
        computation_time+=compf.tv_sec-comps.tv_sec+(compf.tv_usec-comps.tv_usec)*0.000001;


        }   
     }



    gettimeofday(&tf,NULL);
    total_time=tf.tv_sec-ts.tv_sec+(tf.tv_usec-ts.tv_usec)*0.000001;
	
    //Gather local matrices back to the global matrix
    if (rank==0) {
        A=malloc2D(X_ext,Y);    
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
        printf("LU-Block-p2p\tSize\t%d\tProcesses\t%d\n",X,size);
        printf("Max times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",max_total,max_comp,max_comp);
        printf("Avg times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",avg_total,avg_comp,avg_comp);
    }

    //Print triangular matrix U to file
    if (rank==0) {
        char * filename="output_block_p2p";
        print2DFile(A,X,Y,filename);
    }


    MPI_Finalize();

    return 0;
}


