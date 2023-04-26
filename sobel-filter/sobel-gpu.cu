/***********************************************************************
 * sobel-cpu.cu
 *
 * Implements a Sobel filter on the image that is hard-coded in main.
 * You might add the image name as a command line option if you were
 * to use this more than as a one-off assignment.
 *
 * See https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
 * or https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
 * for info on how the filter is implemented.
 *
 * Compile/run with:  nvcc sobel-cpu.cu -lfreeimage
 *
 ***********************************************************************/
#include "FreeImage.h"
#include "stdio.h"
#include "math.h"
#define THREADDIM  64
#define IMAGE_NAME "coings.png"
// Returns the index into the 1d pixel array
// Given te desired x,y, and image width
int pixelIndex(int x, int y, int width)
{
    return (y*width + x);
}

// Returns the sobel value for pixel x,y
__global__ void sobel(int *edge,  char *pixels, int imgwidth, int imgheight)
{
  
   int x00 = -1;  int x20 = 1;
   int x01 = -2;  int x21 = 2;
   int x02 = -1;  int x22 = 1;
   int y = blockIdx.y*blockDim.y+threadIdx.y;
   int x = blockIdx.x*blockDim.x+threadIdx.x; 
//   if(x==0 and y==0){
//	int i=0;
//	while(pixels[i]!=0){
//		i++;
//		printf("%d: %d\n",i,pixels[i]);
//	}
//}
//   __shared__ int pix_data[20];
   if(x>0 and y>0 and x<imgwidth-1 and y<imgheight-1){    
   x00 *= pixels[(x-1)+(y-1)*imgwidth];
   x01 *= pixels[(x-1)+y*imgwidth];
   x02 *= pixels[(x-1)+(y+1)*imgwidth];
   x20 *= pixels[(x+1)+(y-1)*imgwidth];
   x21 *= pixels[(x+1)+y*imgwidth];
   x22 *= pixels[(x+1)+(y+1)*imgwidth];
   
   int y00 = -1;  int y10 = -2;  int y20 = -1;
   int y02 = 1;  int y12 = 2;  int y22 = 1;
   y00 *= pixels[(x-1)+(y-1)*imgwidth];
   y10 *= pixels[x+(y-1)*imgwidth];
   y20 *= pixels[(x+1)+(y-1)*imgwidth];
   y02 *= pixels[(x-1)+(y+1)*imgwidth];
   y12 *= pixels[x+(y+1)*imgwidth];
   y22 *= pixels[(x+1)+(y+1)*imgwidth];

  
   float px = x00 + x01 + x02 + x20 + x21 + x22;
   float py = y00 + y10 + y20 + y02 + y12 + y22;
//    if(x==1 and pixels[x+y*imgwidth]!=-1)
//	printf("%d: %d\n",x+y*imgwidth,pixels[x+y*imgwidth]);//(int)sqrt((px*px)+(py*py)));

   //pixels[x+y*width] = sqrtf(px*px+py*py);
							
   edge[x+y*imgwidth] = (int)sqrt((px*px) + (py*py));
//   if(x==1 and pixels[x+y*imgwidth]!=-1)
//	printf("edge %d: %d\n",x+y*imgwidth,edge[x+y*imgwidth]);
   //atomicExch(&edge[x+y*width],);
 }
}

int main()
{
    FreeImage_Initialise();
    atexit(FreeImage_DeInitialise);

    // Load image and get the width and height
    FIBITMAP *image;
    image = FreeImage_Load(FIF_PNG, IMAGE_NAME, 0);
    if (image == NULL)
    {
        printf("Image Load Problem\n");
        exit(0);
    }
    int imgWidth;
    int imgHeight;
    imgWidth = FreeImage_GetWidth(image);
    imgHeight = FreeImage_GetHeight(image);

    // Convert image into a flat array of chars with the value 0-255 of the
    // greyscale intensity
    RGBQUAD aPixel;
    char *pixels;
    int *edge;
    int *dev_edge;
    char *dev_pix;  
    int pixIndex = 0;
    int exw = 0;
    int exh = 0;
    pixels = (char *) malloc(sizeof(char)*imgWidth*imgHeight);
    edge = (int *) malloc(sizeof(int)*imgWidth*imgHeight);
    for (int i = 0; i < imgHeight; i++)
     for (int j = 0; j < imgWidth; j++)
     {
       FreeImage_GetPixelColor(image,j,i,&aPixel);
       char grey = ((aPixel.rgbRed + aPixel.rgbGreen + aPixel.rgbBlue)/3);
       pixels[pixIndex++]=grey;
     }
    printf("%d, %d, %d, %d\n",imgWidth,imgHeight, imgWidth/THREADDIM, imgWidth%THREADDIM);
    cudaMalloc((void**)&dev_pix, imgWidth*imgHeight*sizeof(char));
    cudaMalloc((void**)&dev_edge, imgWidth*imgHeight*sizeof(int));
    cudaMemcpy(dev_pix, pixels, imgWidth*imgHeight*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_edge, edge, imgWidth*imgHeight*sizeof(int), cudaMemcpyHostToDevice);
	
    if(imgWidth%THREADDIM != 0)
   	exw = 1;
    if(imgHeight%THREADDIM != 0)
	exh = 1;
    printf("%d\n", (imgWidth/THREADDIM)+exw);
    dim3 grid((imgWidth/THREADDIM)+exw, (imgHeight/THREADDIM)+exh);
    dim3 threads(THREADDIM, THREADDIM);    
    sobel<<<grid, threads>>>(dev_edge, dev_pix, imgWidth, imgHeight);
    cudaMemcpy(edge, dev_edge, imgWidth*imgHeight*sizeof(int), cudaMemcpyDeviceToHost);
//    printf("%d\n", strlen(edge));
  

    // Apply sobel operator to pixels, ignoring the borders
   FIBITMAP *bitmap = FreeImage_Allocate(imgWidth, imgHeight, 24);
  for (int i = 1; i < imgWidth-1; i++)
  {
    for (int j = 1; j < imgHeight-1; j++)
    {
	aPixel.rgbRed = edge[i+j*(imgWidth)];
	aPixel.rgbGreen = edge[i+j*(imgWidth)];
	aPixel.rgbBlue = edge[i+j*(imgWidth)];
//	if(j==1 and edge[i+j*imgWidth]!=0)
//		printf("i,j: %d, %d\n",j+i*imgWidth,edge[j+i*imgWidth]);
        FreeImage_SetPixelColor(bitmap, i, j, &aPixel);
      }
    }
   FreeImage_Save(FIF_PNG, bitmap, "output-edge.png", 0);
  
    free(pixels);
    FreeImage_Unload(bitmap);
    FreeImage_Unload(image);
    return 0;
}
