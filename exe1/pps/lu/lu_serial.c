/***** Serial LU decomposition *****/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval ts, tf;

double **allocate(int n,int m)
{
	int i,j;
	double **array;
	double * temp;
	temp=malloc(m*n*sizeof(double));
	array=malloc(n*sizeof(double*));
	for (i=0;i<n;i++)
		array[i]=temp+i*m;	
	for (i=0;i<n;i++)
		for (j=0;j<m;j++) 
			array[i][j]=(double)rand()/1000000.0;	
	return array;
}

void print(double ** a,int N)
{
	int i,j;
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++)
			printf("%.2lf ",a[i][j]);
		printf("\n");
	}
}

int main(int argc, char * argv[])
{
	double ** A, l;
	int N,i,j,k;
	struct timeval ts,tf;
	if (argc<2) {
		printf("Usage: ./[executable] [grid_size]\n");
		exit(1);
	}
	N=atoi(argv[1]);
	A=allocate(N,N);
	gettimeofday(&ts,NULL);
	for (k=0;k<N-1;k++) 
		for (i=k+1;i<N;i++)
		{
			l=A[i][k]=(double)A[i][k]/(double)A[k][k];
			for (j=k+1;j<N;j++)		
			
				A[i][j]-=l*A[k][j];
		}
	gettimeofday(&tf,NULL);
	double time=(tf.tv_sec-ts.tv_sec)+(tf.tv_usec-ts.tv_usec)*0.000001;
	printf("Serial\t%d\t%lf\n",N,time);
//	print(A,N);
	return 0;
}
