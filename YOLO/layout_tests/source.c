
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

void conv2d(float* x, float* w, float* y){
	for (int j = 0; j < 13; j++){
	for (int c_in = 0; c_in < 1024; c_in++){
	for (int i = 0; i < 13; i++){
	for (int dj = 0; dj < 3; dj++){
	for (int di = 0; di < 3; di++){
	for (int c_out = 0; c_out < 1024; c_out++){
		int x_index = (i+di)*13*1024 + (j+dj)*1024 + c_in;
		int y_index = i*13*1024 + j*1024 + c_out;
		int w_index = di*3*1024*1024 + dj*1024*1024 + c_in*1024 + c_out;
		y[y_index] += x[x_index] * w[w_index];
	}}}}}}
}

int main(){
    srand(time(NULL));
    float* x = aligned_alloc(32, 15*15*1024*sizeof(float));
    float* w = aligned_alloc(32, 3*3*1024*1024*sizeof(float));
    float* y = aligned_alloc(32, 13*13*1024*sizeof(float));

    for (int i = 0; i < 15*15*1024; i++){
        x[i] = (float) rand()/ (float) RAND_MAX;
    }
    for (int i = 0; i < 3*3*1024*1024; i++){
        w[i] = (float) rand()/ (float) RAND_MAX;
    }

    clock_t begin = clock();

    conv2d(x, w, y);

    clock_t end = clock();

    unsigned long millis = (end -  begin) * 1000 / CLOCKS_PER_SEC;

    printf( "Finished in %ld ms\n", millis );

}
