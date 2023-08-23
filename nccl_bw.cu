#include <iostream>
#include <nccl.h>

const int kWarmUpTurns = 100;
const size_t kDataSize = 128 * 1024 * 1024; // 128 MB

void checkNcclError(ncclResult_t result, int line) {
    if (result != ncclSuccess) {
        std::cerr << "NCCL Error " << result << " at line " << line << ": " << ncclGetErrorString(result) << std::endl;
        exit(1);
    }
}
#define NCCL_CHECK(cmd) checkNcclError(cmd, __LINE__)

float P2PBandwidthTest(int device_id1, int device_id2) {
    float* sendBuffer;
    float* recvBuffer;
    
    cudaSetDevice(device_id1);
    cudaMalloc(&sendBuffer, kDataSize * sizeof(float));
    cudaMemset(sendBuffer, 1.0, kDataSize * sizeof(float));

    cudaSetDevice(device_id2);
    cudaMalloc(&recvBuffer, kDataSize * sizeof(float));
    cudaMemset(recvBuffer, 0.0, kDataSize * sizeof(float));

    ncclComm_t comm;
    int rank = 0;
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    NCCL_CHECK(ncclCommInitRank(&comm, 2, id, rank));

    for (int i = 0; i < kWarmUpTurns; ++i) {
        NCCL_CHECK(ncclSend(sendBuffer, kDataSize, ncclFloat, device_id2, comm, cudaStreamDefault));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    NCCL_CHECK(ncclSend(sendBuffer, kDataSize, ncclFloat, device_id2, comm, cudaStreamDefault));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    NCCL_CHECK(ncclCommDestroy(comm));

    cudaSetDevice(device_id1);
    cudaFree(sendBuffer);

    cudaSetDevice(device_id2);
    cudaFree(recvBuffer);

    return milliseconds / 1000.0; // Convert to seconds
}

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    if (numGPUs < 2) {
        std::cout << "Error: At least two GPUs are required." << std::endl;
        return 0;
    }

    for (int index1 = 0; index1 < numGPUs; ++index1) {
        for (int index2 = 0; index2 < numGPUs; ++index2) {
            if (index1 == index2) continue; // Skip self-to-self communication

            float time = P2PBandwidthTest(index1, index2);
            float bandwidth = (kDataSize * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / time; // in GB/s
            printf("GPU %d -> GPU %d: Bandwidth: %10.2f GB/s\n", index1, index2, bandwidth);
        }
    }

    return 0;
}