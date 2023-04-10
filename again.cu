#include <stdio.h>
#define COLUMNS 8
#define ROWS  8
const int THREADS_PER_BLOCK = COLUMNS;
const int N = COLUMNS*ROWS;

__global__ void add(int* a, int* c){
	__shared__ int cache[THREADS_PER_BLOCK];
	int tid = threadIdx.x + (blockIdx.x*blockDim.x);
	int cacheIndex = threadIdx.x;
	int temp = 0;
	
	while(tid<N){
		temp += a[tid];
		tid += blockDim.x*gridDim.x;
	}
	cache[cacheIndex] = temp;

	int i = blockDim.x/2;
	while(i>0){
		if(cacheIndex<i)
		cache[cacheIndex] += cache[cacheIndex+i];
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x==0)
		c[blockIdx.x]=cache[0];
	
}
int main(){
	int a[ROWS][COLUMNS], c[COLUMNS], sum=0;
	int *dev_a, * dev_c;
	
	cudaMalloc((void**)&dev_a, ROWS*COLUMNS*sizeof(int));
	cudaMalloc((void**)&dev_c, ROWS*COLUMNS*sizeof(int));

	for(int y=0; y<ROWS; y++){
		for(int x=0; x<COLUMNS; x++){
			a[y][x] = x+y+10;
		}
	}	

	cudaMemcpy(dev_a, a, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
	
	add <<< ROWS, 	COLUMNS>>>(dev_a, dev_c);
	cudaMemcpy(c, dev_c, COLUMNS*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<COLUMNS; i++){
		sum += c[i];
	}
	printf("%d\n", sum);	
	cudaFree(dev_a);
	cudaFree(dev_c);
	return 0;
}
