#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <tbb/task.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>
#include "mm_recursive.h"

matrix newmatrix(int);          /* allocate storage */
void freematrix (matrix, int); /*free storage */
void randomfill(int, matrix);   /* fill with random values in the the range [0,1) */
void auxrandomfill(int, matrix, int, int);
void print (int, matrix, FILE *); /*print matrix in file*/
void auxprint(int, matrix, FILE *, int, int );
void check(int, char *);        /* check for error conditions */
int block;


class RecAddTask : public tbb::task {

    public:
      int n,i,j;
      matrix a,b,c;

      RecAddTask(int n_, matrix a_, matrix b_, matrix c_) : n(n_), a(a_), b(b_), c(c_)  {}

      task* execute() {
        if (n<=block) {
          double **p = a->d, **q = b->d, **r = c->d;
          for (i=0; i<n; i++)
            for (j=0; j<n; j++)
                r[i][j] = p[i][j] + q[i][j];
 
        } else {
  
        n/=2;
        RecAddTask& t1 = *new(tbb::task::allocate_child() ) RecAddTask(n, a11, b11, c11);
        RecAddTask& t2 = *new(tbb::task::allocate_child() ) RecAddTask(n, a12, b12, c12);
        RecAddTask& t3 = *new(tbb::task::allocate_child() ) RecAddTask(n, a21, b21, c21);
        RecAddTask& t4 = *new(tbb::task::allocate_child() ) RecAddTask(n, a12, b22, c22);

        set_ref_count(5);
 
        tbb::task::spawn(t1);
        tbb::task::spawn(t2);
        tbb::task::spawn(t3);
        tbb::task::spawn(t4);
        tbb::task::wait_for_all();     
     }
        return NULL;
    }
};



class RecMultTask : public tbb::task {  

    public:
    int n,jj,kk;
    double temp;
    matrix a,b,c;
    RecMultTask(int n_, matrix a_, matrix b_, matrix c_) : n(n_), a(a_), b(b_), c(c_)  {}


    task* execute() {
      matrix d; 
      if (n<=block) {
        double sum, **p = a->d, **q = b->d, **r = c->d;
        int i, j, k;
/*
        for (i = 0; i < n; i++) {
           for (j = 0; j < n; j++) {
              for (sum = 0., k = 0; k < n; k++)
                  sum += p[i][k] * q[k][j];
                r[i][j] = sum;
               }
          }
*/

     for(int jj=0;jj<n;jj+= 16){
        for(int kk=0; kk<n; kk+= 16){
                for(int i=0;i<n; i++){
                        for(int j = jj; j<((jj+16)>n ? n:(jj+16)); j++){
                                temp = 0;
                                for(int k = kk; k<((kk+16) > n ?n :(kk+16)); k++){
                                        temp += p[i][k]*q[k][j];
                                }
                                r[i][j] += temp;
              }
             }
           }
}



    } else {

      d=newmatrix(n);
      n/=2;
      RecMultTask& t1 = *new(tbb::task::allocate_child() ) RecMultTask(n, a11, b11, d11);
      RecMultTask& t2 = *new(tbb::task::allocate_child() ) RecMultTask(n, a12, b21, c11);
      RecMultTask& t3 = *new(tbb::task::allocate_child() ) RecMultTask(n, a11, b12, d12);
      RecMultTask& t4 = *new(tbb::task::allocate_child() ) RecMultTask(n, a12, b22, c12);
      RecMultTask& t5 = *new(tbb::task::allocate_child() ) RecMultTask(n, a21, b11, d21);
      RecMultTask& t6 = *new(tbb::task::allocate_child() ) RecMultTask(n, a22, b21, c21);
      RecMultTask& t7 = *new(tbb::task::allocate_child() ) RecMultTask(n, a21, b12, d22);
      RecMultTask& t8 = *new(tbb::task::allocate_child() ) RecMultTask(n, a22, b22, c22);     

      set_ref_count(9); 
      
      tbb::task::spawn(t1);
      tbb::task::spawn(t2);
      tbb::task::spawn(t3);
      tbb::task::spawn(t4);
      tbb::task::spawn(t5);
      tbb::task::spawn(t6);
      tbb::task::spawn(t7);
      tbb::task::spawn(t8);    
      tbb::task::wait_for_all();

      RecAddTask& t9  = *new(tbb::task::allocate_child() ) RecAddTask(n, c11, c11, d11);
      RecAddTask& t10 = *new(tbb::task::allocate_child() ) RecAddTask(n, c12, c12, d12);
      RecAddTask& t11 = *new(tbb::task::allocate_child() ) RecAddTask(n, c21, c21, d21);
      RecAddTask& t12 = *new(tbb::task::allocate_child() ) RecAddTask(n, c22, c22, d22);

      set_ref_count(5);

      tbb::task::spawn(t9);
      tbb::task::spawn(t10);
      tbb::task::spawn(t11);
      tbb::task::spawn(t12);
      tbb::task::wait_for_all();
 
   }
      return NULL;
    }  
};


void RecAddTG(int n, matrix a, matrix b, matrix c) {
    if (n <= block) {
        double **p = a->d, **q = b->d, **r = c->d;
        int i, j;
        for (i = 0; i < n; i++) 
            for (j = 0; j < n; j++) 
                r[i][j] = p[i][j] + q[i][j];
    } 
    else {
        n /= 2;
        tbb::task_group g;
        g.run( [&]{ RecAddTG(n, a11, b11, c11); });
        g.run( [&]{ RecAddTG(n, a12, b12, c12); });
        g.run( [&]{ RecAddTG(n, a21, b21, c21); });
        g.run( [&]{ RecAddTG(n, a22, b22, c22); });
        g.wait();
    }
}


void MatrMultTG(int n, matrix a, matrix b, matrix c){

  matrix d;
  if (n<=block) {
   double sum, **p = a->d, **q = b->d, **r = c->d, temp;
   int i, j, k,jj, kk;
   
/*   for (i = 0; i < n; i++) {
     for (j = 0; j < n; j++) {
        for (sum = 0., k = 0; k < n; k++)
           sum += p[i][k] * q[k][j];
           r[i][j] = sum;
            }
       } */

     for(int jj=0;jj<n;jj+= 16){
        for(int kk=0; kk<n; kk+= 16){
                for(int i=0;i<n; i++){
                        for(int j = jj; j<((jj+16)>n ? n:(jj+16)); j++){
                                temp = 0;
                                for(int k = kk; k<((kk+16) > n ?n :(kk+16)); k++){
                                        temp += p[i][k]*q[k][j];
                                }
                                r[i][j] += temp;
              }
             }
           }
}



  } else {
    d=newmatrix(n);
    n/=2;
    tbb::task_group g;
  
    g.run([&] { MatrMultTG(n, a11, b11, d11); });
    g.run([&] { MatrMultTG(n, a12, b21, c11); });
    g.run([&] { MatrMultTG(n, a11, b12, d12); });
    g.run([&] { MatrMultTG(n, a12, b22, c12); });
    g.run([&] { MatrMultTG(n, a21, b11, d21); });
    g.run([&] { MatrMultTG(n, a22, b21, c21); });
    g.run([&] { MatrMultTG(n, a21, b12, d22); });
    g.run([&] { MatrMultTG(n, a22, b22, c22); });
    g.wait();

    g.run([&] { RecAddTG(n, d11, c11, c11); });
    g.run([&] { RecAddTG(n, d12, c12, c12); });
    g.run([&] { RecAddTG(n, d21, c21, c21); });
    g.run([&] { RecAddTG(n, d22, c22, c22); });
    g.wait();

    freematrix(d,n*2);

  }
};




int main(int argc, char* argv[]) {
   
  int nthreads=0;
  int n=0;
  matrix a,b,c;
  tbb::tick_count tic, toc;
  n = atoi(argv[1]);
  block=atoi(argv[2]);
  nthreads=atoi(argv[3]);
  a = newmatrix(n);
  b = newmatrix(n);
  c = newmatrix(n);
  randomfill(n, a);
  randomfill(n, b);
  randomfill(n,c);

  tbb::task_scheduler_init init(nthreads);
  tic = tbb::tick_count::now ();

  RecMultTask &start = *new(tbb::task::allocate_root()) RecMultTask(n, a, b, c);
  tbb::task::spawn_root_and_wait(start);

  toc = tbb::tick_count::now();
  std::cout << (toc - tic).seconds() << "\n";  

  tic = tbb::tick_count::now ();
  MatrMultTG(n,a,b,c);
  toc = tbb::tick_count::now ();
  std::cout << (toc - tic).seconds() << "\n";

  freematrix(a,n);
  freematrix(b,n);
  freematrix(c,n);

return 0;
}




/* fill n by n matrix with random numbers */
void randomfill(int n, matrix a) {
        int i,j;
        double T = -(double)(1 << 31);

        if (n <= block) {
                for (i = 0; i < n; i++)
                        for (j = 0; j < n; j++)
                                a->d[i][j] = rand() / T;
        }
        else {
                for (i=0;i<n;i++)
                        for (j=0;j<n;j++)
                                auxrandomfill(n,a,i,j);
        }
}

void auxrandomfill(int n, matrix a, int i, int j) {

        double T = -(double)(1 << 31);
        if (n<=block)
                a->d[i][j]=rand()/T;
        else
                auxrandomfill(n/2,a->p[(i>=n/2)*2+(j>=n/2)],i%(n/2),j%(n/2));
}

/* return new square n by n matrix */
matrix newmatrix(int n) {

        matrix a;
        a = (matrix)malloc(sizeof(*a));
        check(a != NULL, "newmatrix: out of space for matrix");
        if (n <= block) {
                int i;
                a->d = (double **)calloc(n, sizeof(double *));
                check(a->d != NULL,
                        "newmatrix: out of space for row pointers");
                for (i = 0; i < n; i++) {
                        a->d[i] = (double *)calloc(n, sizeof(double));
                        check(a != NULL, "newmatrix: out of space for rows");
                }
        }
        else {
                n /= 2;
                a->p = (matrix *)calloc(4, sizeof(matrix));
                check(a->p != NULL,"newmatrix: out of space for submatrices");
                a11 = newmatrix(n);
                a12 = newmatrix(n);
                a21 = newmatrix(n);
                a22 = newmatrix(n);
        }
        return a;
}

/* free square n by n matrix m */
void freematrix (matrix m, int n) {
        if (n<=block) {
                int i;
                for (i=0;i<n;i++)
                        free(m->d[i]);
                free(m->d);
        }
        else {
                n/=2;
                freematrix(m->p[0],n);
                freematrix(m->p[1],n);
                freematrix(m->p[2],n);
                freematrix(m->p[3],n);
        }
}

/* print n by n matrix into file f*/
void print(int n, matrix a, FILE * f) {
        int i,j;

        if (n <= block) {
                for (i = 0; i < n; i++) {
                        for (j = 0; j < n; j++)
                                fprintf(f,"%lf ",a->d[i][j]);
                        fprintf(f,"\n");
                }
        }
        else {
                for (i=0;i<n;i++) {
                        for (j=0;j<n;j++)
                                auxprint(n,a,f,i,j);
                        fprintf(f,"\n");
                }
        }
}

void auxprint(int n, matrix a, FILE * f, int i, int j) {

        if (n<=block)
                fprintf(f,"%lf ",a->d[i][j]);
        else
                auxprint(n/2,a->p[(i>=n/2)*2+(j>=n/2)],f,i%(n/2),j%(n/2));
}

/*
 * If the expression e is false print the error message s and quit. 
 */

void check(int e, char *s)
{
    if (!e) {
                fprintf(stderr, "Fatal error -> %s\n", s);
                exit(1);
    }
}

