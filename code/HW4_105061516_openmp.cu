#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

const int INF = 10000000;
const int V = 10010;
void input(char *inFileName);
void output(char *outFileName);

void block_FW();
int ceil(int a, int b);
void callP1(int round);
void callP2(int r, 
int *block_start_x, int *block_start_y, 
int *block_height, int *block_width);
__global__ void cal_Pone(int* Dist_ij);
__global__ void cal_Ptwo(int* Dist_ij, int* Dist_ik, int* Dist_kj);


int n, m, numDevs, B;	// Number of vertices, edges
static int Dist[V][V];
//int* Dist;

int main(int argc, char* argv[])
{
	numDevs = 0; 
	cudaGetDeviceCount(&numDevs); 
	/*
	for (int d = 0; d < numDevs; d++)
		cudaSetDevice(d);
	cudaSetDevice(0);*/
	input(argv[1]);
	B = atoi(argv[3]);
	if(B < 1)
		B = 1;
	else if(B > 32)// B^2 < max thread == 1024 
		B = 32;
	/*
	cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop, 0) == cudaSuccess) 
	{
    	printf("cuda version = %d \n" , prop.major ) ;
		printf("maxThreadsPerBlock = %d \n" , prop.maxThreadsPerBlock ) ;
		printf("totalGlobalMem = %d \n" , prop.totalGlobalMem ) ;
		printf(" maxThreadsDim[3] = %d, %d, %d\n" , prop.maxThreadsDim[1], prop.maxThreadsDim[2] , prop.maxThreadsDim[3] ) ;
		printf(" maxGridSize[3] = %d, %d, %d\n" , prop.maxGridSize[1] , prop.maxGridSize[2] , prop.maxGridSize[3] ) ;
    }
	//cuda version: 2
	//maxThreadsPerBlock: 1024
	//totalGlobalMem: 2066153472
	//maxThreadsDim: 1024, 64, 65535
	//maxGridSize: 65535, 65535, 1301000
	//*/
	block_FW();
	
	
	output(argv[2]);
	

	return 0;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);
	
	

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		Dist[a][b] = v;
	}
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Dist[i][j] >= INF)	fprintf(outfile, "INF ");
			else					fprintf(outfile, "%d ", Dist[i][j]);
		}
		fprintf(outfile, "\n");
	}		
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW()
{
	int round = ceil(n, B);
	
	/* Phase 1*/
	callP1(round);
	
	for (int r = 0; r < round; ++r) {
		/* Phase 2*/
		//printf("This is Phase 2\n");
		int block_start_x[4] = {r,		r, 0,	r + 1};
		int block_start_y[4] = {0,	r + 1, r,		r};
		int block_height[4] = {1, 				1, r,	round - r - 1};
		int block_width[4] = {r,	round - r - 1, 1, 				1};
		callP2(r, block_start_x, block_start_y,
			block_height, block_width);
		/* hase 3*/
		//printf("This is Phase 3\n");
		int block_start_x2[4] = {0, 	0,	r + 1, r + 1};
		int block_start_y2[4] = {0,	r + 1,		0, r + 1};
		int block_height2[4] = {r, r,	round - r -1,	round - r -1};
		int block_width2[4] = {r,	round - r -1, r,	round - r -1};
		callP2(r, block_start_x2, block_start_y2,
			block_height2, block_width2);
		//printf("End one turn\n");
	}

}

void putDistInArray(int round, int bias_x, int bias_y, 
	int block_height, int block_width,
	int* Dist_ij, int* Dist_ik, int* Dist_kj){
		
	int shm_size = sizeof(int) * B * B;
	cudaError_t err = cudaMallocHost(&Dist_ij, shm_size * block_height * block_width);
	if (err != 0)	printf("malloc Dist_ij error\n");
	err = cudaMallocHost(&Dist_ik, shm_size * block_height);
	if (err != 0)	printf("malloc Dist_ik error\n");
	err = cudaMallocHost(&Dist_kj, shm_size * block_width);
	if (err != 0)	printf("malloc Dist_kj error\n");
	int ktmp, itmp, jtmp, j, k;
	int kbias = round * B;
	int end_x = block_height * B;
	int end_y = block_width * B;
	bias_x *= B;
	bias_y *= B;
	//為何要重新歷過一遍?比原本的還慢!!
	//因要解決IO問題，不想直接IO全部
	//真的有比較差??不是被"資源利用率"騙到?(IO全部VS處理完再IO部分)
	for(int i = 0; i < end_x; ++i){
		itmp = bias_x + i;
		if(itmp >= n)
			itmp = n - 1;
		for(j = 0; j < end_y; ++j){
			jtmp = bias_y + j;
			if(jtmp >= n)
				jtmp = n - 1;
			Dist_ij[i * end_y + j] = Dist[itmp][jtmp];
		}
		for(k = 0; k < B; ++k){
			ktmp = k + kbias;
			if(ktmp >= n)
				ktmp = n - 1;
			Dist_ik[i * B + k] = Dist[itmp][ktmp];
		}
	}
	for(int k = 0; k < B; ++k){
		ktmp = k + kbias;
		if(ktmp >= n)
			ktmp = n - 1;
		for(int j = 0; j < end_y; ++j){
			jtmp = bias_y + j;
			if(jtmp >= n)
				jtmp = n - 1;
			Dist_kj[k * end_y + j] = Dist[ktmp][jtmp];
		}
	}
}
void putToDist(int round, int bias_x, int bias_y, 
	int block_height, int block_width,
	int* Dist_ij){
	
	int itmp, jtmp;
	int end_x = block_height * B;
	int end_y = block_width * B;
	bias_x *= B;
	bias_y *= B;
	for(int i = 0; i < end_x; ++i){
		itmp = bias_x + i;
		if(itmp >= n)
			break;
		for(int j = 0; j < end_y; ++j){
			jtmp = bias_y + j;
			if(jtmp >= n)
				break;
			Dist[itmp][jtmp] = Dist_ij[i * end_y + j];
		}
	}
}
void callP1(int round){
	
	int shm_size = sizeof(int) * B * B;
	dim3 blocksPerGrif1(1, 1);
	dim3 threadsPerBlock(B, B);
	
	/*
	round = ceil(n,numDevs *B));
	#pragma omp parallel num_threads(numDevs)
	if (cudaSetDevice(omp_get_thread_num()) == cudaSuccess)
	{
		for(int i = 0; i < round; i++)
		if( i *round *numDevs > n)
			continue;
		dim3 dimBlock(32,16);
		dim3 dimGrid(65535,65535);
		kernel<<<dimGrid,dimBlock>>>();
		cudaThreadExit();
	}
	//*/
	cudaStream_t stream[round];
	int *Dist_all[round * 2];// pointer array
	int i;
	int thread_id[round];
	#pragma omp parallel shared(round, shm_size, blocksPerGrif1, threadsPerBlock, stream, Dist_all, i, thread_id) num_threads(numDevs)
	{
	int itmp, jtmp, bias_y, i2, j;
	cudaError_t err ;
	#pragma omp for schedule(dynamic)
	for(i = 0; i < round; ++i){
		thread_id[i] = omp_get_thread_num();
		cudaSetDevice(thread_id[i]);
		cudaStreamCreate(&stream[i]);
		int *Dist_ij;
		err = cudaMallocHost(&Dist_ij, shm_size * 1 * 1);
		if (err != 0)	printf("malloc Dist_ij error\n");
		bias_y = i * B;
		for(i2 = 0; i2 < B; ++i2){
			itmp = bias_y + i2;
			if(itmp >= n)
				itmp = n - 1;
			for(j = 0; j < B; ++j){
				jtmp = bias_y + j;
				if(jtmp >= n)
					jtmp = n - 1;
				Dist_ij[i2 * B + j] = Dist[itmp][jtmp];
			}
		}
		Dist_all[i *2 ] = Dist_ij;
		int *Dist_ijg;
		
		//step 1: declare
		cudaMalloc((void **)&Dist_ijg, shm_size);
		Dist_all[i *2 +1] = Dist_ijg;
		//step 2: copy
		cudaMemcpyAsync(Dist_ijg, Dist_ij, shm_size, cudaMemcpyHostToDevice, stream[i]);
		
		cal_Pone<<< blocksPerGrif1 , threadsPerBlock , shm_size>>> 
			(Dist_ijg);
		//step 3: get return
		cudaMemcpyAsync(Dist_ij, Dist_ijg, shm_size, cudaMemcpyDeviceToHost, stream[i]);
	}
	}
	//wait for stream
	#pragma omp parallel for shared(round, stream, Dist_all, i) num_threads(numDevs) schedule(dynamic)
	for(i = 0; i < round; i++){
		cudaSetDevice(thread_id[i]);
		cudaStreamSynchronize(stream[i]);
		putToDist(i, i, i,
			1, 1, Dist_all[i *2]);
		//step 4: free gpu
		cudaFree(Dist_all[i *2]);
		cudaFreeHost(Dist_all[i *2 +1]);
			
		cudaStreamDestroy(stream[i]);
		
	}
	
}
void callP2(int r, 
int *block_start_x, int *block_start_y, 
int *block_height, int *block_width){
	
	int shm_size = sizeof(int) * B * B;
	dim3 threadsPerBlock(B, B);
	const int str_num = 4;
	cudaStream_t stream[str_num];
	int *Dist_all[str_num *6];// pointer array
	int kbias = r * B;
	int i;
	int thread_id[str_num];
	#pragma omp parallel shared( shm_size, threadsPerBlock, stream, Dist_all, kbias, i, thread_id) num_threads(numDevs) 
	{
	int ktmp, itmp, jtmp, i2, j, k, bias_x, bias_y, end_x, end_y;
	cudaError_t err;
	#pragma omp for schedule(dynamic)
	for(int i = 0; i < str_num; ++i){
		if( block_height[i] == 0 || block_width[i] == 0)
			continue;
		thread_id[i] = omp_get_thread_num();
		cudaSetDevice(thread_id[i]);
		cudaStreamCreate(&stream[i]);
		dim3 blocksPerGrif1( block_height[i], block_width[i]);
		int *Dist_ij2, *Dist_ik2, *Dist_kj2;
		//putDistInArray(r, B, block_start_x[i], block_start_y[i], 
		//	block_height[i], block_width[i], Dist_ij2, Dist_ik2, Dist_kj2);
		
		err = cudaMallocHost(&Dist_ij2, shm_size * block_height[i] * block_width[i]);
		if (err != 0)	printf("malloc Dist_ij2 error\n");
		err = cudaMallocHost(&Dist_ik2, shm_size * block_height[i]);
		if (err != 0)	printf("malloc Dist_ik2 error\n");
		err = cudaMallocHost(&Dist_kj2, shm_size * block_width[i]);
		if (err != 0)	printf("malloc Dist_kj2 error\n");
		end_x = block_height[i] * B;
		end_y = block_width[i] * B;
		bias_x = block_start_x[i] *B;
		bias_y = block_start_y[i] *B;
		//為何要重新歷過一遍?比原本的還慢!!
		//因要解決IO問題，不想直接IO全部
		//真的有比較差??不是被"資源利用率"騙到?(IO全部VS處理完再IO部分)
		for(i2 = 0; i2 < end_x; ++i2){
			itmp = bias_x + i2;
			if(itmp >= n)
				itmp = n -1;
			for(j = 0; j < end_y; ++j){
				jtmp = bias_y + j;
				if(jtmp >= n)
					jtmp = n -1;
				Dist_ij2[i2 * end_y + j] = Dist[itmp][jtmp];
			}
			for(k = 0; k < B; ++k){
				ktmp = k + kbias;
				if(ktmp >= n)
					ktmp = n -1;
				Dist_ik2[i2 * B + k] = Dist[itmp][ktmp];
			}
		}
		for(int k = 0; k < B; ++k){
			ktmp = k + kbias;
			if(ktmp >= n)
				ktmp = n -1;
			for(int j = 0; j < end_y; ++j){
				jtmp = bias_y + j;
				if(jtmp >= n)
					jtmp = n -1;
				Dist_kj2[k * end_y + j] = Dist[ktmp][jtmp];
			}
		}
	
		Dist_all[i *6] = Dist_ij2;
		Dist_all[i *6 +1] = Dist_ik2;
		Dist_all[i *6 +2] = Dist_kj2;
		int *Dist_ijg2, *Dist_ikg2, *Dist_kjg2;
		//step 1: declare
		cudaMalloc((void **)&Dist_ijg2, shm_size * block_height[i] * block_width[i]);
		cudaMalloc((void **)&Dist_ikg2, shm_size * block_height[i]);
		cudaMalloc((void **)&Dist_kjg2, shm_size * block_width[i]);
		Dist_all[i *6 +3] = Dist_ijg2;
		Dist_all[i *6 +4] = Dist_ikg2;
		Dist_all[i *6 +5] = Dist_kjg2;
		//step 2: copy
		cudaMemcpyAsync(Dist_ijg2, Dist_ij2, shm_size * block_height[i] * block_width[i], 
			cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(Dist_ikg2, Dist_ik2, shm_size * block_height[i], 
			cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(Dist_kjg2, Dist_kj2, shm_size * block_width[i], 
			cudaMemcpyHostToDevice, stream[i]);
	
		cal_Ptwo<<< blocksPerGrif1 , threadsPerBlock , shm_size, stream[i]>>> 
			(Dist_ijg2, Dist_ikg2, Dist_kjg2);
		//step 3: get return
		cudaMemcpyAsync(Dist_ij2, Dist_ijg2, shm_size * block_height[i] * block_width[i], 
			cudaMemcpyDeviceToHost, stream[i]);
	}
	}
	//wait for stream
	// private(  j, tt2 )
	#pragma omp parallel for shared(block_start_x, block_start_y, block_height, block_width, stream, Dist_all, i, thread_id) num_threads(numDevs) schedule(dynamic)
	for(int i = 0; i < str_num; i++){
		if( block_height[i] == 0 || block_width[i] == 0)
			continue;
		cudaSetDevice(thread_id[i]);
		cudaStreamSynchronize(stream[i]);
		putToDist(r, block_start_x[i], block_start_y[i],
			block_height[i], block_width[i], Dist_all[i *6]);
		//step 4: free gpu
		cudaFree(Dist_all[i *6 +3]);
		cudaFree(Dist_all[i *6 +4]);
		cudaFree(Dist_all[i *6 +5]);
		cudaFreeHost(Dist_all[i *6]);
		cudaFreeHost(Dist_all[i *6 +1]);
		cudaFreeHost(Dist_all[i *6 +2]);
			
		cudaStreamDestroy(stream[i]);
		
	}
}
__global__ void cal_Pone(int* Dist_ij)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ int DS[];
	int dsbias = threadIdx.x * blockDim.y + threadIdx.y;
	int offset_j = gridDim.y * blockDim.y;
	DS[dsbias] = Dist_ij[i * offset_j + j];//j range = blocksPerG.y
	__syncthreads();
	for (int k = 0; k < blockDim.x ; ++k) {//k range= B
		if (DS[i * blockDim.x + k] + DS[k * offset_j + j] < DS[dsbias])
			DS[dsbias] = DS[i * blockDim.x + k] + DS[k * offset_j + j];
		__syncthreads();
	}
	Dist_ij[i * offset_j + j] = DS[dsbias];// save value from shared memory
	__syncthreads();
	
}
__global__ void cal_Ptwo(int* Dist_ij, int* Dist_ik, int* Dist_kj)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ int DS[];
	int dsbias = threadIdx.x * blockDim.y + threadIdx.y;
	int offset_j = gridDim.y * blockDim.y;
	DS[dsbias] = Dist_ij[i * offset_j + j];//j range = blocksPerG.y
	__syncthreads();
	for (int k = 0; k < blockDim.x ; ++k) {//k range= B
		if (Dist_ik[i * blockDim.x + k] + Dist_kj[k * offset_j + j] < DS[dsbias])
			DS[dsbias] = Dist_ik[i * blockDim.x + k] + Dist_kj[k * offset_j + j];
	}
	Dist_ij[i * offset_j + j] = DS[dsbias];// save value to shared memory
	__syncthreads();
}

/*
void cal(int B, int Round, 
	int block_start_x, int block_start_y, 
	int block_width, int block_height)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				int block_internal_start_x 	= b_i * B;
				int block_internal_end_x 	= (b_i +1) * B;
				int block_internal_start_y = b_j * B; 
				int block_internal_end_y 	= (b_j +1) * B;

				if (block_internal_end_x > n)	block_internal_end_x = n;
				if (block_internal_end_y > n)	block_internal_end_y = n;

				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
						if (Dist[i * n + k] + Dist[k * n + j] < Dist[i * n + j])
							Dist[i * n + j] = Dist[i * n + k] + Dist[k * n + j];
					}
				}
			}
		}
	}
}
*/

