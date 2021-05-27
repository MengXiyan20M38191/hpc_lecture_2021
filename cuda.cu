#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

#define TILE_SIZE 32

__global__ void matrixMul(const float* A, const float* B, float* C, int N)
{
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Col = bx * TILE_SIZE + tx;
    int Row = by * TILE_SIZE + ty;

    float sum = 0.0f;
    
    for(int i = 0; i < N / TILE_SIZE; i++)
    {
        s_A[ty][tx] = A[Row * N + (i * TILE_SIZE + tx)];
        s_B[ty][tx] = B[Col + (i * TILE_SIZE + ty) * N];
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++)
            sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }

    C[Row * N + Col] = sum;
}

int main(int argc, char** argv) {
    const int N = 256;
    vector<float> A(N*N);
    vector<float> B(N*N);
    vector<float> C(N*N, 0);

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            A[N*i+j] = drand48();
            B[N*i+j] = drand48();
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * N);

    cudaMemcpy(d_A, &A[0], sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &B[0], sizeof(float) * N * N, cudaMemcpyHostToDevice);
    auto tic = chrono::steady_clock::now();

    int GRID_SIZE = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    matrixMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto toc = chrono::steady_clock::now();
    double comp_time = chrono::duration<double>(toc - tic).count();

    cudaMemcpy(&C[0], d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            for (int k=0; k<N; k++)
                C[N*i+j] -= A[N*i+k] * B[N*k+j];

    double err = 0;
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            err += fabs(C[i*N+j]);

    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("total: %lf s (%lf GFlops)\n", comp_time, 2.*N*N*N/comp_time/1e9);
    printf("error: %lf\n",err/N/N);
}
