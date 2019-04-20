#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <random>

typedef unsigned long long uint64;

#pragma region CUDA_CONSTANTS

__constant__ int PC_1[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

__constant__ int PC_2[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

__constant__ int IP[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

__constant__ int E_BIT[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

__constant__ int S1[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

__constant__ int S2[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

__constant__ int S3[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

__constant__ int S4[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

__constant__ int S5[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

__constant__ int S6[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

__constant__ int S7[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

__constant__ int S8[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

__constant__ int* ALL_S[8] = {
	S1, S2, S3, S4, S5, S6, S7, S8
};

__constant__ int P[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

__constant__ int IP_REV[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

__constant__ int SHIFTS[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};

#pragma endregion

#pragma region HOST_CONSTANTS

int PC_1_HOST[56] = {
	57,	49,	41,	33,	25,	17,	9,
	1,	58,	50,	42,	34,	26,	18,
	10,	2,	59,	51,	43,	35,	27,
	19,	11,	3,	60,	52,	44,	36,
	63,	55,	47,	39,	31,	23,	15,
	7,	62,	54,	46,	38,	30,	22,
	14,	6,	61,	53,	45,	37,	29,
	21,	13,	5,	28,	20,	12,	4
};

int PC_2_HOST[48] = {
	14, 17, 11, 24, 1,	5,
	3,	28, 15, 6,	21, 10,
	23, 19, 12, 4,	26, 8,
	16, 7,	27, 20, 13, 2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};

int IP_HOST[64] = {
	58,	50,	42,	34,	26,	18,	10,	2,
	60,	52,	44,	36,	28,	20,	12,	4,
	62,	54,	46,	38,	30,	22,	14,	6,
	64,	56,	48,	40,	32,	24,	16,	8,
	57,	49,	41,	33,	25,	17,	 9,	1,
	59,	51,	43,	35,	27,	19,	11,	3,
	61,	53,	45,	37,	29,	21,	13,	5,
	63,	55,	47,	39,	31,	23,	15,	7
};

int E_BIT_HOST[48] = {
	32,	1,	2,	3,	4,	5,
	4,	5,	6,	7,	8,	9,
	8,	9,	10,	11,	12,	13,
	12,	13,	14,	15,	16,	17,
	16,	17,	18,	19,	20,	21,
	20,	21,	22,	23,	24,	25,
	24,	25,	26,	27,	28,	29,
	28,	29,	30,	31,	32,	1
};

int S1_HOST[64] = {
	14,	4,	13,	1,	2,	15,	11,	8,	3,	10,	6,	12,	5,	9,	0,	7,
	0,	15,	7,	4,	14,	2,	13,	1,	10,	6,	12,	11,	9,	5,	3,	8,
	4,	1,	14,	8,	13,	6,	2,	11,	15,	12,	9,	7,	3,	10,	5,	0,
	15,	12,	8,	2,	4,	9,	1,	7,	5,	11,	3,	14,	10,	0,	6,	13
};

int S2_HOST[64] = {
	15,	1,	8,	14,	6,	11,	3,	4,	9,	7,	2,	13,	12,	0,	5,	10,
	3,	13,	4,	7,	15,	2,	8,	14,	12,	0,	1,	10,	6,	9,	11,	5,
	0,	14,	7,	11,	10,	4,	13,	1,	5,	8,	12,	6,	9,	3,	2,	15,
	13,	8,	10,	1,	3,	15,	4,	2,	11,	6,	7,	12,	0,	5,	14,	9,
};

int S3_HOST[64] = {
	10,	0,	9,	14,	6,	3,	15,	5,	1,	13,	12,	7,	11,	4,	2,	8,
	13,	7,	0,	9,	3,	4,	6,	10,	2,	8,	5,	14,	12,	11,	15,	1,
	13,	6,	4,	9,	8,	15,	3,	0,	11,	1,	2,	12,	5,	10,	14,	7,
	1,	10,	13,	0,	6,	9,	8,	7,	4,	15,	14,	3,	11,	5,	2,	12
};

int S4_HOST[64] = {
	7,	13,	14,	3,	0,	6,	9,	10,	1,	2,	8,	5,	11,	12,	4,	15,
	13,	8,	11,	5,	6,	15,	0,	3,	4,	7,	2,	12,	1,	10,	14,	9,
	10,	6,	9,	0,	12,	11,	7,	13,	15,	1,	3,	14,	5,	2,	8,	4,
	3,	15,	0,	6,	10,	1,	13,	8,	9,	4,	5,	11,	12,	7,	2,	14
};

int S5_HOST[64] = {
	2,	12,	4,	1,	7,	10,	11,	6,	8,	5,	3,	15,	13,	0,	14,	9,
	14,	11,	2,	12,	4,	7,	13,	1,	5,	0,	15,	10,	3,	9,	8,	6,
	4,	2,	1,	11,	10,	13,	7,	8,	15,	9,	12,	5,	6,	3,	0,	14,
	11,	8,	12,	7,	1,	14,	2,	13,	6,	15,	0,	9,	10,	4,	5,	3
};

int S6_HOST[64] = {
	12,	1,	10,	15,	9,	2,	6,	8,	0,	13,	3,	4,	14,	7,	5,	11,
	10,	15,	4,	2,	7,	12,	9,	5,	6,	1,	13,	14,	0,	11,	3,	8,
	9,	14,	15,	5,	2,	8,	12,	3,	7,	0,	4,	10,	1,	13,	11,	6,
	4,	3,	2,	12,	9,	5,	15,	10,	11,	14,	1,	7,	6,	0,	8,	13,
};

int S7_HOST[64] = {
	4,	11,	2,	14,	15,	0,	8,	13,	3,	12,	9,	7,	5,	10,	6,	1,
	13,	0,	11,	7,	4,	9,	1,	10,	14,	3,	5,	12,	2,	15,	8,	6,
	1,	4,	11,	13,	12,	3,	7,	14,	10,	15,	6,	8,	0,	5,	9,	2,
	6,	11,	13,	8,	1,	4,	10,	7,	9,	5,	0,	15,	14,	2,	3,	12,
};

int S8_HOST[64] = {
	13,	2,	8,	4,	6,	15,	11,	1,	10,	9,	3,	14,	5,	0,	12,	7,
	1,	15,	13,	8,	10,	3,	7,	4,	12,	5,	6,	11,	0,	14,	9,	2,
	7,	11,	4,	1,	9,	12,	14,	2,	0,	6,	10,	13,	15,	3,	5,	8,
	2,	1,	14,	7,	4,	10,	8,	13,	15,	12,	9,	0,	3,	5,	6,	11,
};

int* ALL_S_HOST[8] = {
	S1_HOST, S2_HOST, S3_HOST, S4_HOST, S5_HOST, S6_HOST, S7_HOST, S8_HOST
};

int P_HOST[32] = {
	16,	7,	20, 21,
	29,	12, 28, 17,
	1,	15, 23, 26,
	5,	18, 31, 10,
	2,	8,	24, 14,
	32, 27, 3,	9,
	19, 13, 30,	6,
	22, 11, 4,	25
};

int IP_REV_HOST[64] = {
	40,	8, 48, 16, 56, 24, 64, 32,
	39, 7, 47, 15, 55, 23, 63, 31,
	38, 6, 46, 14, 54, 22, 62, 30,
	37, 5, 45, 13, 53, 21, 61, 29,
	36, 4, 44, 12, 52, 20, 60, 28,
	35, 3, 43, 11, 51, 19, 59, 27,
	34, 2, 42, 10, 50, 18, 58, 26,
	33, 1, 41,	9, 49, 17, 57, 25
};

int SHIFTS_HOST[16] = {
	1,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1,
	2,
	2,
	2,
	2,
	2,
	2,
	1
};

#pragma endregion

#pragma region CUDA_WRAPPERS

void cudaSetDeviceWrapper()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
	}
}

void cudaMallocWrapper(void **destination, size_t size)
{
	cudaError_t cudaStatus = cudaMalloc(destination, size);
	if (cudaStatus != cudaSuccess) {
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
	}
}

void cudaDeviceSynchronizeWrapper()
{
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
	}
}

void cudaMemcpyWrapper(void* destination, void* source, size_t size, cudaMemcpyKind memcpyKind)
{
	cudaError_t cudaStatus = cudaMemcpy(destination, source, size, memcpyKind);
	if (cudaStatus != cudaSuccess) {
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(1);
	}
}

#pragma endregion

__device__ __host__ uint64 GetBit(uint64 number, int bitNumber);
__device__ __host__ void SetBit(uint64* number, int bitNumber, uint64 valueToSet);
__device__ __host__ uint64 CycleBitsLeftSide(uint64 value, int numberOfPlacesToShift, int bitsInValue);
__device__ __host__ uint64 Permute(uint64 key, int* permutationToApply, int lenght, int keyLenght);
__device__ __host__ void SplitInHalf(uint64 key, uint64* left, uint64* right, int keyLenght);
__host__ uint64 GenerateDesKey(int keyLenght);
__device__  uint64 EncryptData(uint64 dataToEncrypt, uint64 desKey);
__device__  void CreateSubKeys(uint64* subKeys, uint64 desKey);
__device__ uint64 F(uint64 data, uint64 key);
__device__ uint64 Encode(uint64* subKeys, uint64 dataToEncrypt);
__host__  uint64 EncryptDataHost(uint64 dataToEncrypt, uint64 desKey);
__host__  void CreateSubKeysHost(uint64* subKeys, uint64 desKey);
__host__ uint64 FHost(uint64 data, uint64 key);
__host__ uint64 EncodeHost(uint64* subKeys, uint64 dataToEncrypt);
__global__ void Crack(uint64 data, uint64 encodedData, uint64 *key, bool *done, uint64 maxLenght);


int main()
{
	cudaSetDeviceWrapper();
	std::cout << "Enter key lenght:" << std::endl;

	int keyLenght;
	std::cin >> keyLenght;
	int maxLenght = 1 << keyLenght;

	uint64 desKey = GenerateDesKey(keyLenght);
	uint64 dataToEncrypt = 0x0123456789ABCDEF;
	uint64 encryptedData = EncryptDataHost(dataToEncrypt, desKey);

	uint64* devKey = NULL, crackedKey;
	int done_val = 0;
	bool *done = NULL;
	cudaMallocWrapper((void**)&devKey, sizeof(uint64));
	cudaMallocWrapper((void**)&done, sizeof(int));
	cudaMemcpyWrapper(done, &done_val, sizeof(int), cudaMemcpyHostToDevice);

	std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
	Crack << <4096, 1024 >> > (dataToEncrypt, encryptedData, devKey, done, maxLenght);

	cudaDeviceSynchronizeWrapper();
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	auto gpuExecutionTime = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0;

	cudaMemcpyWrapper(&crackedKey, devKey, sizeof(uint64), cudaMemcpyDeviceToHost);
	uint64 encryptedDataWithKeyFromGPU = EncryptDataHost(dataToEncrypt, crackedKey);
	if (encryptedDataWithKeyFromGPU == encryptedData)
	{
		std::cout << "[GPU] Found matching key in " << gpuExecutionTime << " seconds" << std::endl;
		std::cout << "Found key: " << crackedKey << std::endl;
		std::cout << "Original key: " << desKey << std::endl << std::endl;
	}
	else if (crackedKey == 0)
	{
		std::cout << "[GPU] Can not find matching key!" << std::endl << std::endl;
	}
	else
	{
		std::cout << "[GPU] Found key do not work!" << std::endl;
	}


	begin = std::chrono::system_clock::now();
	int keyFound = -1;
	for (uint64 i = 0; i < maxLenght; i++)
	{
		uint64 currentValue = EncryptDataHost(dataToEncrypt, i);
		if (currentValue == encryptedData)
		{
			keyFound = i;
			break;
		}
	}
	end = std::chrono::system_clock::now();
	auto cpuExecutionTime = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0;

	if (keyFound != -1)
	{
		std::cout << "[CPU] Found matching key in " << cpuExecutionTime << " sekund" << std::endl;
		std::cout << "Found key: " << keyFound << std::endl;
		std::cout << "Original key: " << desKey << std::endl;
	}
	else
	{
		std::cout << "[CPU] Can not find matching key!" << std::endl;
	}

	std::cout << "GPU solving time is " << gpuExecutionTime / cpuExecutionTime * 100 << " % CPU solving time." << std::endl;

	cudaFree(devKey);
	cudaFree(done);
}

__global__ void Crack(uint64 data, uint64 encodedData, uint64 *key, bool *foundKey, uint64 maxLenght)
{
	for (uint64 i = blockIdx.x * blockDim.x + threadIdx.x; i <= maxLenght; i += blockDim.x * gridDim.x)
	{
		uint64 currentValue = EncryptData(data, i);
		if (currentValue == encodedData)
		{
			*key = i;
			*foundKey = false;
			return;
		}
		if (*foundKey == true)
		{
			return;
		}
	}
}

#pragma region DeviceAndHostFunctions

__device__ __host__ uint64 GetBit(uint64 number, int bitNumber)
{
	return 1ULL & (number >> bitNumber);
}

__device__ __host__ void SetBit(uint64* number, int bitNumber, uint64 valueToSet)
{
	*number = *number & ~(1ULL << bitNumber) | (valueToSet << bitNumber);
	//*number ^= (-valueToSet ^ *number) & (1ULL << bitNumber);
}

__device__ __host__ uint64 CycleBitsLeftSide(uint64 value, int numberOfPlacesToShift, int bitsInValue)
{
	for (int i = 0; i < numberOfPlacesToShift; i++)
	{
		uint64 bit = GetBit(value, bitsInValue - 1);
		value <<= 1;
		SetBit(&value, bitsInValue, 0);
		SetBit(&value, 0, bit);
	}
	return value;
}

__device__ __host__ uint64 Permute(uint64 key, int* permutationToApply, int lenght, int keyLenght)
{
	uint64 permutedKey = 0;
	for (int i = 0; i < lenght; i++)
	{
		SetBit(&permutedKey, i, GetBit(key, keyLenght - permutationToApply[lenght - i - 1]));
	}
	return permutedKey;
}

__device__ __host__ void SplitInHalf(uint64 key, uint64* left, uint64* right, int keyLenght)
{
	*right = *left = 0;
	for (int i = 0; i < keyLenght / 2; i++)
	{
		SetBit(left, i, GetBit(key, keyLenght / 2 + i));
		SetBit(right, i, GetBit(key, i));
	}
}

#pragma endregion

#pragma region DeviceFunctions

__device__ uint64 EncryptData(uint64 dataToEncrypt, uint64 desKey)
{
	uint64 subKeys[16];
	CreateSubKeys(subKeys, desKey);
	return Encode(subKeys, dataToEncrypt);
}

__device__  void CreateSubKeys(uint64* subKeys, uint64 desKey)
{
	// K+
	uint64 keyPlus = Permute(desKey, PC_1, 56, 64);

	// Cn, Dn
	uint64_t C[17];
	uint64_t D[17];

	// C0, D0
	SplitInHalf(keyPlus, &C[0], &D[0], 56);

	// C1-C16, D1-D16
	for (int i = 1; i <= 16; i++)
	{
		C[i] = CycleBitsLeftSide(C[i - 1], SHIFTS[i - 1], 28);
		D[i] = CycleBitsLeftSide(D[i - 1], SHIFTS[i - 1], 28);
	}

	// K1-K16
	for (int i = 0; i <= 15; i++)
	{
		subKeys[i] = C[i + 1] << 28 | D[i + 1];
		subKeys[i] = Permute(subKeys[i], PC_2, 48, 56);
	}
}

__device__ uint64 F(uint64 data, uint64 key)
{
	uint64 expand = Permute(data, E_BIT, 48, 32);
	uint64 xor = expand ^ key;

	uint64 S[8];
	uint64 B[8];

	for (int i = 0; i < 8; i++)
	{
		B[i] = 0;

		// Fill B[i]
		for (int j = 0; j < 6; j++)
		{
			SetBit(&B[i], j, GetBit(xor, j + (7 - i) * 6));
		}

		uint64 firstAndLastBit = GetBit(B[i], 5) << 1 | GetBit(B[i], 0);
		uint64 middleBits = GetBit(B[i], 4) << 3 | GetBit(B[i], 3) << 2 | GetBit(B[i], 2) << 1 | GetBit(B[i], 1);

		// Calculate S[i]
		S[i] = ALL_S[i][(int)firstAndLastBit * 16 + (int)middleBits];
	}

	// Combine results
	uint64 result = 0;

	for (int i = 0; i < 8; i++)
	{
		result |= S[i] << 28 - 4 * i;
	}

	// Apply final P permutation
	return Permute(result, P, 32, 32);
}

__device__  uint64 Encode(uint64* subKeys, uint64 dataToEncrypt)
{
	uint64 ip = Permute(dataToEncrypt, IP, 64, 64);

	uint64 left, right;
	SplitInHalf(ip, &left, &right, 64);

	for (int i = 0; i < 16; i++)
	{
		uint64 prevLeft = left, prevRight = right;
		left = right;
		right = prevLeft ^ F(prevRight, subKeys[i]);
	}

	uint64 leftWithRightReversed = right << 32 | left;

	// Final IP^(-1) permutation
	return Permute(leftWithRightReversed, IP_REV, 64, 64);
}

#pragma endregion

#pragma region HostFunctions

__host__ uint64 GenerateDesKey(int keyLenght)
{
	std::mt19937 mt;
	std::uniform_int_distribution<int> randomBinary(0, 1);

	uint64 key = 0;
	for (int i = 0; i < keyLenght; i++)
	{
		SetBit(&key, i, randomBinary(mt));
	}
	return key;
}

__host__ uint64 EncryptDataHost(uint64 dataToEncrypt, uint64 desKey)
{
	uint64 subKeys[16];
	CreateSubKeysHost(subKeys, desKey);
	return EncodeHost(subKeys, dataToEncrypt);
}

// Step 1
__host__ void CreateSubKeysHost(uint64* subKeys, uint64 desKey)
{
	// K+
	uint64 keyPlus = Permute(desKey, PC_1_HOST, 56, 64);

	// Cn, Dn
	uint64_t C[17];
	uint64_t D[17];

	// C0, D0
	SplitInHalf(keyPlus, &C[0], &D[0], 56);

	// C1-C16, D1-D16
	for (int i = 1; i <= 16; i++)
	{
		C[i] = CycleBitsLeftSide(C[i - 1], SHIFTS_HOST[i - 1], 28);
		D[i] = CycleBitsLeftSide(D[i - 1], SHIFTS_HOST[i - 1], 28);
	}

	// K1-K16
	for (int i = 0; i <= 15; i++)
	{
		subKeys[i] = C[i + 1] << 28 | D[i + 1];
		subKeys[i] = Permute(subKeys[i], PC_2_HOST, 48, 56);
	}
}

__host__ uint64 FHost(uint64 data, uint64 key)
{
	uint64 expand = Permute(data, E_BIT_HOST, 48, 32);
	uint64 xor = expand ^ key;

	uint64 S[8];
	uint64 B[8];
	for (int i = 0; i < 8; i++)
	{
		B[i] = 0;

		// Fill B[i]
		for (int j = 0; j < 6; j++)
		{
			SetBit(&B[i], j, GetBit(xor, j + (7 - i) * 6));
		}

		uint64 firstAndLastBit = GetBit(B[i], 5) << 1 | GetBit(B[i], 0);
		uint64 middleBits = GetBit(B[i], 4) << 3 | GetBit(B[i], 3) << 2 | GetBit(B[i], 2) << 1 | GetBit(B[i], 1);

		// Calculate S[i]
		S[i] = ALL_S_HOST[i][(int)firstAndLastBit * 16 + (int)middleBits];
	}

	// Combine results
	uint64 result = 0;
	for (int i = 0; i < 8; i++)
	{
		result |= S[i] << 28 - 4 * i;
	}

	// Apply final P permutation
	return Permute(result, P_HOST, 32, 32);
}

__host__ uint64 EncodeHost(uint64* subKeys, uint64 dataToEncrypt)
{
	uint64 ip = Permute(dataToEncrypt, IP_HOST, 64, 64);

	uint64 left, right;
	SplitInHalf(ip, &left, &right, 64);

	for (int i = 0; i < 16; i++)
	{
		uint64 prevLeft = left, prevRight = right;
		left = right;
		right = prevLeft ^ FHost(prevRight, subKeys[i]);
	}

	uint64 leftWithRightReversed = right << 32 | left;

	// Final IP^(-1) permutation
	return Permute(leftWithRightReversed, IP_REV_HOST, 64, 64);
}

#pragma endregion