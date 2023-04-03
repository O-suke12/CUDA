#include <stdio.h>
#define COLUMNS 4
#define ROWS  3

__global__ void add(int* a, int* c){
	int x = threadIdx.x;
	int sum = 0;

	for (int i=0; i<ROWS; i++)
		sum += a[i*COLUMNS+x];
	c[x] = sum;	
}
int main(){
	int a[ROWS][COLUMNS], c[COLUMNS], sum=0;
	int *dev_a, * dev_c;
	
	cudaMalloc((void**)&dev_a, ROWS*COLUMNS*sizeof(int));
	cudaMalloc((void**)&dev_c, ROWS*COLUMNS*sizeof(int));

	for(int y=0; y<ROWS; y++){
		for(int x=0; x<COLUMNS; x++){
			a[y][x] = x+y;
		}
	}	

	cudaMemcpy(dev_a, a, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
	
	add <<< 1, 	COLUMNS>>>(dev_a, dev_c);
	cudaMemcpy(c, dev_c, COLUMNS*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<COLUMNS; i++){
		sum += c[i];
	}
	printf("%d\n",sum);
}
