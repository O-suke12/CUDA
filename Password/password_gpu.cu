
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "md5.cu"
#define GRIDDIM 550
#define BLOCKDIM 32

// Convert a decimal number (starting at 0) to a corresponding 6 letter string
// using base 26 to represent the string
// s must be big enough to hold 6 chars plus a null (7 chars total)
__device__ void intToString(int num, char *s)
{
  int ones = (num) % 26;
  int twentySix = (num / 26) % 26;
  int twentySixSquared = (num / 26 / 26) % 26;
  int twentySixCubed = (num / 26 / 26 / 26) % 26;
  int twentySixFourth = (num / 26 / 26 / 26 / 26) % 26;
  int twentySixFifth = (num / 26 / 26 / 26 / 26 / 26) % 26;
  // Store appropriate char into the string
  int i = 0;
  s[i++] = twentySixFifth + 'A';
  s[i++] = twentySixFourth + 'A';
  s[i++] = twentySixCubed + 'A';
  s[i++] = twentySixSquared + 'A';
  s[i++] = twentySix + 'A';
  s[i++] = ones + 'A';
  s[i] = '\0';
}

// You may find this helpful for testing, this takes a 6 char string
// like ABACAB and returns back the decimal number that maps to it
// using the intToString function above
__device__ int stringToInt(char *s)
{
  unsigned int count = 0;
    while(*s!='\0')
    {
        count++;
        s++;
    }
  int length = (int)count;   
  int sum = 0;
  int power = 0;

  for (int i = length-1; i >= 0; i--)
  {
	int digit = s[i] - 'A';
	sum += digit * pow(26,power);	
	power++;
  } 
  return sum;
}


__global__ void search(int* target){
  char possibleKey[7];  // Will be auto-generated AAAAAA to ZZZZZZ
  uint32_t hashResult1, hashResult2, hashResult3, hashResult4;
  uint8_t length = 6;

  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int i = y*blockDim.x*gridDim.x+x;
  
	intToString(i, possibleKey); 
	md5Hash((unsigned char *) possibleKey, length, 
		&hashResult1, &hashResult2, &hashResult3, &hashResult4);
	if ((hashResult1 == target[0]) &&
            (hashResult2 == target[1]) &&
            (hashResult3 == target[2]) &&
            (hashResult4 == target[3]))
	{ 
		printf("CRACKED! The original string is: %s\n", possibleKey);
    asm("trap;");
    
	}	
	
	if (i % 250000 == 0)
	{
		printf("Guess #%d was %s\n", i, possibleKey);
	}
}


int main(void){
  // This is the md5 hash string we are trying to crack
  char md5_hash_string[] = "070d912366b1cf46a01aaf93c99f907d";
  // char md5_hash_string[] = "b381ce33225e82f0fd839d610c3832e5";
  int md5Target[4];  // The md5 hash string extracted into four integers

  // This loop extracts the md5 hash string into md5Target[0],[1],[2],[3]
  for(int i = 0; i < 4; i++)
  {
    char tmp[16];
    strncpy(tmp, md5_hash_string + i * 8, 8);
    sscanf(tmp, "%x", &md5Target[i]);
    md5Target[i] = (md5Target[i] & 0xFF000000) >> 24 | (md5Target[i] &
                 0x00FF0000) >> 8 | (md5Target[i] & 0x0000FF00) << 8 |
                (md5Target[i] & 0x000000FF) << 24;
  }

  printf("Working on cracking the md5 key %s by trying all key combinations...\n",md5_hash_string);

  int* dev_target;
  cudaMalloc((void**)&dev_target, 4*sizeof(int));
  cudaMemcpy(dev_target, md5Target, 4*sizeof(int), cudaMemcpyHostToDevice);
  dim3 grid(GRIDDIM, GRIDDIM);
  dim3 threads(BLOCKDIM, BLOCKDIM);
  search <<< grid, threads >>> (dev_target);
  cudaDeviceSynchronize();
}