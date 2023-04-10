// This is a simple ray tracer that shoots rays top down toward randomly
// generates spheres and draws the sphere in a random color based on where
// the ray hits it.

#include "FreeImage.h"
#include "stdio.h"

#define DIM 16*128
#define THREADS 16
#define BLOCKS 128
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define THREAD 2

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    // Tells us if a ray hits the sphere; return the
    // depth of the hit, or -infinity if the ray misses the sphere
   __device__  float hit( float ox, float oy, float *n ) 
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius)
        {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

#define SPHERES 80

// Loops through each pixel in the image (represented by arrays of
// red, green, and blue) and then for each pixel checks if a ray from
// top down hits one of the randomly generated spheres.
// If so, calculate a shade of color based on where the ray hits it.
__global__ void drawSpheres(Sphere spheres[], char *red, char *green, char *blue)
{
 int row = blockIdx.y*blockDim.x+threadIdx.y;
 int col = blockIdx.x*blockDim.x+threadIdx.x;
   
 float   ox = (col - DIM/2);
 float   oy = (row - DIM/2);
 float   r=0, g=0, b=0;
 float   maxz = -INF;

for (int j=0; j<SPHERES;j++){
 float   n;
 float   t = spheres[j].hit( ox, oy, &n );
 if (t > maxz)
 {
		// Scale RGB color based on z depth of sphere
        float fscale = n;
        r = spheres[j].r * fscale;
  	g = spheres[j].g * fscale;
        b = spheres[j].b * fscale;
        maxz = t;
 }

    	int offset = col + row * DIM;
    	red[offset] = (char) (r * 255);
    	green[offset] = (char) (g * 255);
    	blue[offset] = (char) (b * 255);

}
} 

int main()
{
  FreeImage_Initialise();
  atexit(FreeImage_DeInitialise);
  FIBITMAP * bitmap = FreeImage_Allocate(DIM, DIM, 24);
  srand(time(NULL));

  char *red=(char*)malloc(DIM*DIM*sizeof(char));
  char *green=(char*)malloc(DIM*DIM*sizeof(char));
  char *blue=(char*)malloc(DIM*DIM*sizeof(char));
  char *dev_r, *dev_g, *dev_b;

  // Dynamically create enough memory for DIM * DIM array of char.
  // By making these dynamic rather than auto (e.g. char red[DIM][DIM])
  // we can make them much bigger since they are allocated off the heap

 cudaMalloc((void**)&dev_r, DIM*DIM*sizeof(char));
 cudaMalloc((void**)&dev_g, DIM*DIM*sizeof(char));
 cudaMalloc((void**)&dev_b, DIM*DIM*sizeof(char));
 cudaMemcpy(dev_r, red, DIM*DIM*sizeof(char), cudaMemcpyHostToDevice);
 cudaMemcpy(dev_g, green, DIM*DIM*sizeof(char), cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, blue, DIM*DIM*sizeof(char), cudaMemcpyHostToDevice);

  // Create random spheres at different coordinates, colors, radius
  Sphere spheres[SPHERES];
  Sphere *dev_sph;
  for (int i = 0; i<SPHERES; i++)
  {
        spheres[i].r = rnd( 1.0f );
        spheres[i].g = rnd( 1.0f );
        spheres[i].b = rnd( 1.0f );
        spheres[i].x = rnd( (float) DIM ) - (DIM/2.0);
        spheres[i].y = rnd( (float) DIM ) - (DIM/2.0);
        spheres[i].z = rnd( (float) DIM ) - (DIM/2.0);
        spheres[i].radius = rnd( 200.0f ) + 40;
  }
 dim3 grid(BLOCKS, BLOCKS);
 dim3 threads(THREADS,THREADS);
 cudaMalloc((void**)&dev_sph, SPHERES*sizeof(struct Sphere));
 cudaMemcpy(dev_sph, spheres, SPHERES*sizeof(struct Sphere), cudaMemcpyHostToDevice);
 drawSpheres<<<grid, threads>>>(dev_sph, dev_r, dev_g, dev_b);	
 cudaMemcpy(red, dev_r, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
 cudaMemcpy(blue, dev_b, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);
 cudaMemcpy(green, dev_g, DIM*DIM*sizeof(char), cudaMemcpyDeviceToHost);

  RGBQUAD color;
  for (int i = 0; i < DIM; i++)
  {
    for (int j = 0; j < DIM; j++)
    {
      int index = j*DIM + i;
      color.rgbRed = red[index];
      color.rgbGreen = green[index];
      color.rgbBlue = blue[index];
      FreeImage_SetPixelColor(bitmap, i, j, &color);
    }
  }
	
  FreeImage_Save(FIF_PNG, bitmap, "ray.png", 0);
  FreeImage_Unload(bitmap);
  free(red);
  free(green);
  free(blue);

  return 0;
}

