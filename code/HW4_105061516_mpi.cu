#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

const int INF = 10000000;
//const int V = 10010;
void input(char *inFileName);
void output(char *outFileName);

void block_FW();
int ceil(int a, int b);
void callP1(int round);
void callP2(int r, 
int *block_start_x, int *block_start_y, 
int *block_height, int *block_width, int phase);
__global__ void cal_Pone(int* Dist_ij);
__global__ void cal_Ptwo(int* Dist_ij, int* Dist_ik, int* Dist_kj);


int n, m, B;	// Number of vertices, edges
int rank, size;
//static int Dist[V *V];
int *Dist;
MPI_Win win;
//int* Dist;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//rank = ver_id - 1
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	cudaSetDevice(rank%2);
	if(rank ==0)
		input(argv[1]);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(rank ==0){
		MPI_Win_create(Dist, n *n *sizeof(int), sizeof(int), MPI_INFO_NULL,    
			MPI_COMM_WORLD, &win);
		//MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), win);
		//MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
		
	}
	else{
		//printf("[%d] n: %d\n", rank, n);
		
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);// MPI_BOTTOM
		/* it works!!
		int tt = 0;
		int tt2 = 0;
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		MPI_Get(&tt, 1, MPI_INT, 0, rank, 1, MPI_INT, win);
		MPI_Get(&tt2, 1, MPI_INT, 0, n, 1, MPI_INT, win);
		MPI_Win_unlock(0, win);
		printf("[%d] get: %d, %d\n", rank, tt, tt2);
		//*/
		/* useless
		MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
		MPI_Win_fence((MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), win);
		//*/
	}
	
	
	B = atoi(argv[3]);
	if(B < 1)
		B = 1;
	else if(B > 32)// B^2 < max thread == 1024 
		B = 32;
	block_FW();
	
	//printf("[%d]This is out BF\n", rank);
	if(rank == 0)
		output(argv[2]);
	

	//MPI_Barrier( MPI_COMM_WORLD );
	MPI_Win_fence(0, win);
	//printf("[%d]This MPI_Win_free\n", rank);
	MPI_Win_free(&win);
	//printf("[%d]This if(rank ==0)\n", rank);
	if(rank ==0)
		MPI_Free_mem(Dist);
	//printf("[%d]This MPI_Finalize\n", rank);
	MPI_Finalize();
	return 0;
}

void input(char *inFileName)
{
	
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);
	MPI_Alloc_mem(n *n * sizeof(int), MPI_INFO_NULL, &Dist);
	

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i *n +j] = 0;
			else		Dist[i *n +j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		Dist[a *n +b] = v;
	}
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Dist[i *n +j] >= INF)	fprintf(outfile, "INF ");
			else					fprintf(outfile, "%d ", Dist[i *n +j]);
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
	
	//MPI_Barrier(MPI_COMM_WORLD);
	int r;
	for(r = 0; r < round; ++r){
		/* Phase 1*/
		if(rank ==0)
			callP1(r);
		/* Phase 2*/
		//printf("[%d][%d]This is Phase 2\n", rank, r);
		int block_start_x[4] = {r,		r, 0,	r + 1};
		int block_start_y[4] = {0,	r + 1, r,		r};
		int block_height[4] = {1, 				1, r,	round - r - 1};
		int block_width[4] = {r,	round - r - 1, 1, 				1};
		callP2(r, block_start_x, block_start_y,
			block_height, block_width, 2);
		/* hase 3*/
		//printf("[%d][%d]This is Phase 3\n", rank, r);
		int block_start_x2[4] = {0, 	0,	r + 1, r + 1};
		int block_start_y2[4] = {0,	r + 1,		0, r + 1};
		int block_height2[4] = {r, r,	round - r -1,	round - r -1};
		int block_width2[4] = {r,	round - r -1, r,	round - r -1};
		callP2(r, block_start_x2, block_start_y2,
			block_height2, block_width2, 3);
		//printf("End one turn\n");
		MPI_Barrier(MPI_COMM_WORLD);
	}
	///* Phase 1*/
	//callP1(round);

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
			Dist_ij[i * end_y + j] = Dist[itmp *n +jtmp];
		}
		for(k = 0; k < B; ++k){
			ktmp = k + kbias;
			if(ktmp >= n)
				ktmp = n - 1;
			Dist_ik[i * B + k] = Dist[itmp *n +ktmp];
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
			Dist_kj[k * end_y + j] = Dist[ktmp *n +jtmp];
		}
	}
}
void putInAij(int bias_y, int* Dist_ij){
		
	int i, j;
	bias_y *= B;
	//為何要重新歷過一遍?比原本的還慢!!
	//因要解決IO問題，不想直接IO全部
	//真的有比較差??不是被"資源利用率"騙到?(IO全部VS處理完再IO部分)

	int jlen = B;
	int itmp ;
	int jtmp ;
	// b+j > n
	if(bias_y + jlen > n)
		jlen = n - bias_y;
	// part 1: <, <
	if(rank ==0){
		for(i = 0; i < jlen; ++i){
			itmp = (bias_y + i)*n +bias_y;
			for(j = 0; j < jlen; ++j)
				Dist_ij[i * B + j] = Dist[itmp +j];
		}
	}
	else{
		for(i = 0; i < jlen; ++i){
			itmp = (bias_y + i) *n +bias_y;
			MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
			MPI_Get(Dist_ij + i *B, jlen, MPI_INT, 0, itmp , jlen, MPI_INT, win);
			MPI_Win_unlock(0, win);
		}
	}
	if( jlen == bias_y)
		return;
	itmp = (jlen -1) * B;
	// part 3: >, <=
	for(i = jlen; i < B; ++i)
		for(j = 0; j < jlen; ++j)
			Dist_ij[i * B + j] = Dist_ij[itmp + j];
	// part 2: >, >
	itmp += jlen -1;//(jlen -1) *B + (jlen -1);
	for(i = jlen; i < B; ++i)
		for(j = jlen; j < B; ++j)
			Dist_ij[i * B + j] = Dist_ij[itmp];
	// part 4: <=, >
	jtmp = (jlen -1);
	for(i = 0; i < jlen; ++i){
		itmp = i *B;
		for(j = jlen; j < B; ++j){
			Dist_ij[itmp + j] = Dist_ij[itmp + jtmp];
		}
	}
}
void putDistInArray_new(int round, int bias_x, int bias_y, 
	int block_height, int block_width,
	int* Dist_ij2, int* Dist_ik2, int* Dist_kj2){
		
	int itmp, jtmp, ktmp, i2, j, k;
	int kbias = round * B;
	int end_x = block_height * B;
	int end_y = block_width * B;
	bias_x *= B;
	bias_y *= B;
	//為何要重新歷過一遍?比原本的還慢!!
	//因要解決IO問題，不想直接IO全部
	//真的有比較差??不是被"資源利用率"騙到?(IO全部VS處理完再IO部分)
	int ilen = end_x;
	int jlen = end_y;
	int klen = B;
	// b+i2 > n
	if(bias_x + ilen > n)
		ilen = n - bias_x;
	// b+j > n
	if(bias_y + jlen > n)
		jlen = n - bias_y;
	// b+j > n
	if(kbias + klen > n)
		klen = n - kbias;
	
	// part 1: <, <, <
	if(rank ==0){
		for(i2 = 0; i2 < ilen; ++i2){
			itmp = (bias_x + i2) *n +bias_y;
			for(j = 0; j < jlen; ++j)
				Dist_ij2[i2 * end_y + j] = Dist[itmp + j];
			itmp += kbias-bias_y;
			for(k = 0; k < klen; ++k)
				Dist_ik2[i2 * B + k] = Dist[itmp +k];
		}
		// part 5: <, <, <
		for(k = 0; k < klen; ++k){
			ktmp = (kbias + k) *n +bias_y;
			for(j = 0; j < jlen; ++j)
				Dist_kj2[k * end_y + j] = Dist[ktmp + j];
		}
	}
	else{
		for(i2 = 0; i2 < ilen; ++i2){
			itmp = (bias_x + i2) *n;
			MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
			MPI_Get(Dist_ij2 + i2 * end_y, jlen, MPI_INT, 0, itmp +bias_y, jlen, MPI_INT, win);
			MPI_Get(Dist_ik2 + i2 *		B, klen, MPI_INT, 0, itmp + kbias, klen, MPI_INT, win);
			MPI_Win_unlock(0, win);
		}
		// part 5: <, <, <
		for(k = 0; k < klen; ++k){
			ktmp = (kbias + k) *n +bias_y;
			MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
			MPI_Get(Dist_kj2 + k  * end_y, jlen, MPI_INT, 0, 		ktmp, jlen, MPI_INT, win);
			MPI_Win_unlock(0, win);
		}
	}
	if(ilen == end_x && jlen == end_y && klen == B )
		return;
	// part 5-2: <, X, > ||  X, <, >
	ktmp = klen -1;
	jtmp = ktmp * end_y;
	for(k = klen; k < B; ++k){
		for(i2 = 0; i2 < ilen; ++i2){
			itmp = i2 * B;
			Dist_ik2[itmp+ k] = Dist_ik2[itmp + ktmp];
		}
		for(j = 0; j < jlen; ++j)
			Dist_kj2[k * end_y + j] = Dist_kj2[jtmp + j];
	}
	// part 3: >, <, X
	itmp = (ilen -1) *end_y;
	for(i2 = ilen; i2 < end_x; ++i2)
		for(j = 0; j < jlen; ++j)
			Dist_ij2[i2 * end_y + j] = Dist_ij2[itmp + j];
	// part 2: >, >, X
	itmp += jlen -1;//(ilen -1) * end_y + (jlen -1);
	for(i2 = ilen; i2 < end_x; ++i2)
		for(j = jlen; j < end_y; ++j)
			Dist_ij2[i2 * end_y + j] = Dist_ij2[itmp];
	
	// part 5-3: >, X, > || X, >, >
	itmp = (ilen -1) * B + ktmp;
	jtmp += jlen -1;//(klen -1) * end_y + (jlen -1);
	for(k = klen; k < B; ++k){
		for(i2 = ilen; i2 < end_x; ++i2)
			Dist_ik2[i2 * B + k] = Dist_ik2[itmp];
		for(j = jlen; j < end_y; ++j)
			Dist_kj2[k * end_y + j] = Dist_kj2[jtmp];
	}
	// part 5-4: >, X, < || X, <, <
	itmp -= ktmp;//(ilen -1) * B ;
	jtmp = jlen -1;
	for(k = 0; k < klen; ++k){
		ktmp = k * end_y;
		for(i2 = ilen; i2 < end_x; ++i2)
			Dist_ik2[i2 * B + k] = Dist_ik2[itmp + k];
		for(j = jlen; j < end_y; ++j)
			Dist_kj2[ktmp + j] = Dist_kj2[ktmp + jtmp];
	}
	// part 4: <, >, X
	for(i2 = 0; i2 < ilen; ++i2){
		itmp = i2 * end_y;
		for(j = jlen; j < end_y; ++j)
			Dist_ij2[itmp + j] = Dist_ij2[itmp + jtmp];
	}
		
	
}


void putToDist(int round, int bias_x, int bias_y, 
	int block_height, int block_width,
	int* Dist_ij){
	
	int itmp, jtmp;
	int end_x = block_height * B;
	int end_y = block_width * B;
	int ilen = end_x;
	int jlen = end_y;
	bias_x *= B;
	bias_y *= B;
	if(end_x + bias_x > n)
		ilen = n - bias_x;
	if(end_y + bias_y > n)
		jlen = n - bias_y;
	if(rank==0){
		for(int i = 0; i < ilen; ++i){
			itmp = bias_x + i;
			for(int j = 0; j < jlen; ++j){
				jtmp = bias_y + j;
				Dist[itmp *n +jtmp] = Dist_ij[i * end_y + j];
			}
		}
	}
	else{
		for(int i = 0; i < ilen; ++i){
			itmp = (bias_x + i)*n + bias_y ;
			MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);// MPI_LOCK_EXCLUSIVE
			MPI_Put(Dist_ij + i *end_y, jlen, MPI_INT, 0, itmp , jlen, MPI_INT, win);
			MPI_Win_unlock(0, win);
		}
	}
}
void callP1(int w_r){
	
	int shm_size = sizeof(int) * B * B;
	dim3 blocksPerGrif1(1, 1);
	dim3 threadsPerBlock(B, B);
	
	cudaError_t err;
	int *Dist_ij;
	err = cudaMallocHost(&Dist_ij, shm_size );
	if (err != 0)	printf("[%d]malloc Dist_ij error\n", rank);
	putInAij(w_r, Dist_ij);
	int *Dist_ijg;
	
	//step 1: declare
	cudaMalloc((void **)&Dist_ijg, shm_size);
	//step 2: copy
	cudaMemcpy(Dist_ijg, Dist_ij, shm_size, cudaMemcpyHostToDevice);
	
	cal_Pone<<< blocksPerGrif1 , threadsPerBlock , shm_size>>> 
		(Dist_ijg);
	//step 3: get return
	cudaMemcpy(Dist_ij, Dist_ijg, shm_size, cudaMemcpyDeviceToHost);
	
	putToDist(w_r, w_r, w_r,
		1, 1, Dist_ij);
	//step 4: free gpu
	cudaFree(Dist_ijg);
	cudaFreeHost(Dist_ij);
}
void callP2(int r, 
int *block_start_x, int *block_start_y, 
int *block_height, int *block_width, int phase){
	
	int shm_size = sizeof(int) * B * B;
	dim3 threadsPerBlock(B, B);
	const int str_num = 4;
	cudaStream_t stream[str_num];
	int *Dist_all[str_num *6];// pointer array
	cudaError_t err;
	int i;
	int ibias = size;
	//*
	if(phase == 3 &&size==2){
		if(rank==0)
			ibias++;
		else
			ibias--;
	}
	//*/
	for(i = rank; i < str_num; i += ibias){
		if( block_height[i] == 0 || block_width[i] == 0 || i > 3)
			continue;
		cudaStreamCreate(&stream[i]);
		dim3 blocksPerGrif1( block_height[i], block_width[i]);
		int *Dist_ij2, *Dist_ik2, *Dist_kj2;
		
		err = cudaMallocHost(&Dist_ij2, shm_size * block_height[i] * block_width[i]);
		if (err != 0)	printf("[%d]malloc Dist_ij2 error\n", rank);
		err = cudaMallocHost(&Dist_ik2, shm_size * block_height[i]);
		if (err != 0)	printf("[%d]malloc Dist_ik2 error\n", rank);
		err = cudaMallocHost(&Dist_kj2, shm_size * block_width[i]);
		if (err != 0)	printf("[%d]malloc Dist_kj2 error\n", rank);
		putDistInArray_new(r, block_start_x[i], block_start_y[i], 
			block_height[i], block_width[i], Dist_ij2, Dist_ik2, Dist_kj2);
		/*
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
				Dist_ij2[i2 * end_y + j] = Dist[itmp *n +jtmp];
			}
			for(k = 0; k < B; ++k){
				ktmp = k + kbias;
				if(ktmp >= n)
					ktmp = n -1;
				Dist_ik2[i2 * B + k] = Dist[itmp *n +ktmp];
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
				Dist_kj2[k * end_y + j] = Dist[ktmp *n +jtmp];
			}
		}
		*/
	
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
	//wait for stream
	for(i = rank; i < str_num; i += ibias){
		if( block_height[i] == 0 || block_width[i] == 0 || i > 3)
			continue;
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

