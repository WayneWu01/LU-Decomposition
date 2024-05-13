#include <iostream>
#include <numa.h>
#include <omp.h>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <sys/time.h>
// int value = 0;

// long fib(int n)
// {
//   if (n < 2) return n;
//   else return fib(n-1) + fib(n-2);
// }

// void usage(const char *name)
// {
// 	std::cout << "usage: " << name
//                   << " matrix-size nworkers"
//                   << std::endl;
//  	exit(-1);
// }
//generate the matrix A
void generate(int n, int num_threads, double** &A){
  A = (double **) malloc(n * sizeof(double *));
  //make it parallel
  #pragma omp parallel num_threads(num_threads)
  {
    struct drand48_data rn;
    srand48_r(omp_get_thread_num() + time(NULL), &rn);
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
      A[i] = (double *) malloc(n * sizeof(double));
      for (int j = 0; j < n; ++j) {
        //random number use drand48
        double rand;
        drand48_r(&rn, &rand);
        A[i][j] = rand * 100;
      }
    }
  }
}
struct Compare {
  double val;
  int index;
};
//LU decomposition method
void decomposition(int n, double** &A, double** &L, double** &U, int* &P) {
  P = new int[n];
  L = new double*[n];
  U = new double*[n];
  //Initialize L and U
  #pragma omp parallel for 
  for (int i = 0; i < n; ++i) {
    P[i] = i;
    L[i] = new double[n];
    U[i] = new double[n];
    memset(L[i], 0, n * sizeof(double));
    memset(U[i], 0, n * sizeof(double));
    L[i][i] = 1.0;
  }
  for (int k = 0; k < n; ++k) {
    Compare max = {0.0, k};
    // Find the maximum element in column
    for (int i = k; i < n; ++i) {
      double val = std::abs(A[i][k]);
      if (val > max.val) {
        max.val = val;
        max.index = i;
      }
    }
    // Check for singularity
    if (max.val == 0) {
      std::cerr << "Singular Matrix" << std::endl;
    }
    std::swap(P[k], P[max.index]);
    std::swap(A[k], A[max.index]);
    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
      std::swap(L[k][i], L[max.index][i]);
    }
    //Elimination step
    #pragma omp parallel for shared(A, L, n, k)
    for (int i = k + 1; i < n; ++i) {
      L[i][k] = A[i][k] / A[k][k];
      A[i][k] = 0.0; 
      for (int j = k + 1; j < n; ++j) {
        A[i][j] -= L[i][k] * A[k][j];
      }
    }
    //Update U accordingly
    #pragma omp parallel for shared(A, U, n, k)
    for (int j = k; j < n; ++j) {
      U[k][j] = A[k][j];
    }
  }
}
//print matrix for checking
void pri(double** matrix, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}
//verify whether L and U are correct or not
double verification(int n, double** A, double** L, double** U, int* P) {
  double norm = 0.0; 
  for (int j = 0; j < n; ++j) {
    double cols = 0.0;
    for (int i = 0; i < n; ++i) {
      double res = 0.0;
      for (int k = 0; k < n; ++k) {
        res += L[i][k] * U[k][j];
      }
      res = A[P[i]][j] - res;
      cols += res * res;
      }
    norm += sqrt(cols);
  }
  return norm; 
}
int main(int argc, const char* argv[]){
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix size> <num threads>" << std::endl;
    return -1;    
  }
  //get parameters
  int n, num_threads;
  n = atoi(argv[1]);
  num_threads = atoi(argv[2]);
  omp_set_num_threads(num_threads);
  //Initialization
  double** A;
  generate(n, num_threads, A);
  //Important :for checking if needeed copy the A to A_copy since A is modified
  // double** A_copy = (double **) malloc(n * sizeof(double *));
  // for (int i = 0; i < n; ++i) {
  //   A_copy[i] = (double *) malloc(n * sizeof(double));
  //   memcpy(A_copy[i], A[i], n * sizeof(double));
  // }
  double** L;
  double** U;
  int* P;
  //Time the decomposition steps
  struct timeval start, end;
  gettimeofday(&start,NULL);
  decomposition(n,A,L,U,P);
  gettimeofday(&end, NULL);
  long double diff = ((end.tv_sec - start.tv_sec) * 1000000.0L + end.tv_usec - start.tv_usec) / 1000000.0L;

  //pri(A, n);
  //pri(L, n);
  //pri(U, n);
  //pri(A, n);
  std::cout << "Decomposition time: " << diff << std::endl; 
  //Perform verification to check the result
  //double res = verification(n,A_copy,L,U,P);
  //std::cout << "Verification result: " << res << std::endl;
}