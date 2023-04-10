#include <stdio.h>
#define ROWS 8
#define COLUMNS 1000000

__global__ void minfind(int*a, int*c)
{
	int min = 1000000001;
	int x = threadIdx.x;
	for(int i=0; i<COLUMNS; i++){
		if(a[x*COLUMNS+i]<min){
			min=a[x*COLUMNS+i];
		}
	}
	c[x]=min;
}

int main(){
	int* dev_a,* dev_c, * a=(int*)malloc(ROWS*COLUMNS*sizeof(int));
	int  c[ROWS], seq_min=1000000000, min=1000000000;
	
	cudaMalloc((void**)&dev_a, ROWS*COLUMNS*sizeof(int));
	cudaMalloc((void**)&dev_c, 8*sizeof(int));
	
	for (int y=0; y<ROWS; y++){
		for (int x=0; x<COLUMNS; x++){
			a[y*COLUMNS+x] = rand()%1000000001;
		}
		cudaMemcpy(dev_a, a, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
	}
	minfind <<<1, ROWS>>>(dev_a,dev_c);
	cudaMemcpy(c, dev_c, 8*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i<ROWS; i++)
		for(int j=0; j<COLUMNS; j++)
			if(a[i*COLUMNS+j]<seq_min)
				seq_min=a[i*COLUMNS+j];
 
	for (int i=0; i<ROWS; i++)
		if(c[i]<min)
			min=c[i];
	printf("Sequential search: %d, Thread search: %d\n",seq_min,min);
	
	cudaFree(dev_a); cudaFree(dev_c);
	return 0;
}

