#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <stdbool.h>
#include <omp.h>


#define CIN 1024
#define COUT 1024
#define H 13
#define K 3


void conv2d(float* x, float* w, float* y){
    
    __m256 zero = _mm256_set1_ps(0);
    for (int i = 0; i < H; i++){
        for (int j = 0; j < H; j++){
            for (int c_out = 0; c_out < COUT; c_out+=8){
                _mm256_store_ps(&(y[i*H*COUT + j*COUT + c_out]), zero);
            }
        }
    }
    
    #pragma omp parallel for // je n'arrive pas à améliorer la perf avec ça
    for (int i = 0; i < H; i++){
        __m256 a, b, c;
        for (int c_in = 0; c_in < CIN; c_in++){
            for (int di = 0; di < K; di++){
                for (int j = 0; j < H; j++){                
                    for (int dj = 0; dj < K; dj++){            
                        b = _mm256_set1_ps(x[(i+di)*(H+2)*CIN + (j+dj)*(H+2) + c_in]);
                        #pragma GCC unroll 16
                        for (int c_out = 0; c_out < COUT; c_out++){
                            
                            a = _mm256_load_ps(&(y[i*H*COUT + j*COUT + c_out]));    
                            
                            c = _mm256_load_ps(&(w[c_in * K*K*COUT + di * K *COUT + dj*COUT + c_out]));

                            c = _mm256_fmadd_ps(c,b,a);
                            //c = _mm256_mul_ps(b,c);
                            //c = _mm256_add_ps(c,a);
                        
                            _mm256_store_ps(&(y[i*H*COUT + j*COUT + c_out]), c);
                            
                        }
                    }
                }
            }
        }
    }
}


int main(){
    srand(time(NULL));
    float* x = aligned_alloc(32, (H+2)*(H+2)*CIN*sizeof(float));
    float* w = aligned_alloc(32, K*K*COUT*CIN*sizeof(float));
    float* y = aligned_alloc(32, H*H*COUT*sizeof(float));

    for (int i = 0; i < (H+2)*(H+2)*CIN; i++){
        x[i] = (float) rand()/ (float) RAND_MAX;
    }
    for (int i = 0; i < K*K*CIN*COUT; i++){
        w[i] = (float) rand()/ (float) RAND_MAX;
    }

    clock_t begin = clock();

    conv2d(x, w, y);

    clock_t end = clock();

    unsigned long millis = (end -  begin) * 1000 / CLOCKS_PER_SEC;

    printf( "Finished in %ld ms\n", millis );
}