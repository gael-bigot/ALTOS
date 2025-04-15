#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <stdbool.h>
#include <pthread.h>


#define CIN 1024
#define COUT 1024
#define H 13
#define K 3


struct thread_params {
    float* x;
    float* w;
    float* y;
    int i;
};

void* thread_func(void* args){
    struct thread_params params = *(struct thread_params*) args;
    float* y = params.y;
    float* x = params.x;
    float* w = params.w;
    int i  = params.i;
    
    __m256 a;
    __m256 b;
    __m256 c;

    for (int c_in = 0; c_in < CIN; c_in++){
        for (int di = 0; di < K; di++){
            for (int j = 0; j < H; j++){                
                for (int dj = 0; dj < K; dj++){            
                    b = _mm256_set1_ps(x[(i+di)*15*CIN + (j+dj)*15 + c_in]);
                    
                    for (int c_out = 0; c_out < COUT; c_out+=8){
                            a = _mm256_load_ps(&(y[i*H*COUT + j*COUT + c_out]));    
                        
                            c = _mm256_load_ps(&(w[c_in * K*K*COUT + di * K *COUT + dj*COUT + c_out]));
                            c = _mm256_mul_ps(c, b);
                            a = _mm256_add_ps(a, c);
                            _mm256_store_ps(&(y[i*H*COUT + j*COUT + c_out]), a);
                    }
                }
            }
        }
    }

    pthread_exit(NULL);
}


void conv2d(float* x, float* w, float* y){
    pthread_t threads[13];
    struct thread_params params[13];
    for (int i = 0; i < 13; i++){
        params[i].x = x;
        params[i].w = w;
        params[i].y = y;
        params[i].i = i;
        pthread_create(&threads[i], NULL, thread_func, &params[i]);
    }
    for (int i = 0; i < 13; i++){
        pthread_join(threads[i], NULL);
    }
}


int main(){
    srand(time(NULL));
    float* x = aligned_alloc(32, 15*15*CIN*sizeof(float));
    float* w = aligned_alloc(32, K*K*CIN*COUT*sizeof(float));
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