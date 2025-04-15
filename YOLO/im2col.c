#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cblas.h>
#include <immintrin.h>

#define H 15
#define C_IN 1024
#define C_OUT 1024

void im2col(float* x, float* y){
    int n, k_i, k_j, di, dj;
    for (int i = 0; i < 9*C_IN; i++){
        for (int j = 0; j < (H-2)*(H-2); j++){
            n = i / 9;
            k_i = j / (H-2);
            k_j = j % (H-2);
            di = (i%9)/3;
            dj = i%3;
            y[i*(H-2)*(H-2) + j] = x[n*H*H+(k_i+di)*H + k_j + dj];
        }
    }
}

void matmul(int M, int N, int K, float* A, int lda, float* B, int ldb, float* C, int ldc){
    __m256 a;
    __m256 b;
    __m256 c;

    int n = N/8;


    for (int i = 0; i < M; i++){
        for (int k = 0; k < K; k++){
            
            a = _mm256_set1_ps(A[i*K+k]);
            
            for (int j = 0; j < n; j++){
                c = _mm256_load_ps(&(C[i*ldc+j*8]));
                b = _mm256_load_ps(&(B[k*ldb+j*8]));
                c = _mm256_fmadd_ps(a,b,c);
                _mm256_store_ps(&(C[i*ldc+j*8]), c);
            }
            
            for (int j = 8*n; j < N; j++){
                C[i*N+j] += A[i*lda+k]*B[k*ldb+j];
            }
        }
    }
}

int main(){
    int M = C_OUT;
    int N = (H-2)*(H-2);
    int N_aligned = N;
    if (N_aligned%32){N_aligned = (N/32+1)*32;}
    int K = C_IN*9;
    int K_aligned = K;
    float* x = aligned_alloc(32, H*H*C_IN*sizeof(float));
    float* x_ = aligned_alloc(32, N_aligned*K*sizeof(float));
    float* w = aligned_alloc(32, M*K*sizeof(float));
    float* y = aligned_alloc(32, M*N_aligned*sizeof(float));

    for (int i = 0; i < H*H*C_IN; i++){
        x[i] = (float) rand() / (float) RAND_MAX;
    }
    for (int i = 0; i < M*K; i++){
        w[i] = (float) rand() / (float) RAND_MAX;
    }
    
    clock_t begin = clock();

    im2col(x,x_);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, w, K, x_, N_aligned, 0.0, y, N_aligned);
    //matmul(M,N,K, w, K, x_, N_aligned, y, N_aligned);

    clock_t end = clock();

    unsigned long millis = (end -  begin) * 1000 / CLOCKS_PER_SEC;

    printf( "Finished in %ld ms\n", millis );

    free(x);
    free(x_);
    free(w);
    free(y);

    return 0;
}