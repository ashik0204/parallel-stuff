// devinfo.cu -- print the GPU properties used in the report.
#include <stdio.h>
#include <cuda_runtime.h>
int main(void)
{
	cudaDeviceProp p;
	if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("No CUDA device\n"); return 1; }
	int memclk = 0;
	cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0); // kHz
	printf("name,%s\n", p.name);
	printf("compute_capability,%d.%d\n", p.major, p.minor);
	printf("sms,%d\n", p.multiProcessorCount);
	printf("max_threads_per_block,%d\n", p.maxThreadsPerBlock);
	printf("max_threads_per_sm,%d\n", p.maxThreadsPerMultiProcessor);
	printf("max_blocks_per_sm,%d\n", p.maxBlocksPerMultiProcessor);
	printf("max_resident_threads,%d\n", p.maxThreadsPerMultiProcessor * p.multiProcessorCount);
	printf("warp_size,%d\n", p.warpSize);
	printf("global_mem_mb,%zu\n", p.totalGlobalMem >> 20);
	printf("shared_mem_per_block_kb,%zu\n", p.sharedMemPerBlock >> 10);
	printf("shared_mem_per_sm_kb,%zu\n", p.sharedMemPerMultiprocessor >> 10);
	printf("l2_cache_kb,%d\n", p.l2CacheSize >> 10);
	printf("mem_bus_width_bits,%d\n", p.memoryBusWidth);
	printf("mem_clock_mhz,%d\n", memclk / 1000);
	printf("peak_bandwidth_gbs,%.1f\n", 2.0 * memclk * 1e3 * (p.memoryBusWidth / 8) / 1e9);
	return 0;
}
