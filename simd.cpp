 
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
using namespace std;

int main() {
  int N = 256;
  float A[N][N], B[N][N], C[N][N];
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }


  auto tic = chrono::steady_clock::now();
  float x[N];
  const int Dig=8;
  float B1[N][N];
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      B1[i][j] = B[j][i];
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k+=Dig) {  //For N = 256 and Digit = 8, Times = N/Dig = 32
        __m256 avec = _mm256_load_ps(*(A+i)+k);
        __m256 bvec = _mm256_load_ps(*(B1+j)+k);
        __m256 cvec = _mm256_mul_ps(avec,bvec);
        __m256 dvec = _mm256_permute2f128_ps(cvec,cvec,1);
        dvec = _mm256_add_ps(dvec,cvec);
        dvec = _mm256_hadd_ps(dvec,dvec);
        dvec = _mm256_hadd_ps(dvec,dvec);
        _mm256_store_ps(x, dvec);
        C[i][j] += x[0];
      }
   }
  }
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf(" N=%d:total= %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);

  float err = 0;
 // for (int i=0; i<N; i++)
   // for (int j=0; j<N; j++)
//      err += fabs(C[i][j]-A[i][j]*B[i][j]);
   for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[i][j]-=(A[i][k]*B[k][j]);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
        err+=fabs(C[i][j]);
  printf("Error=%lf\n",err/N/N);
}
