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

float P2PBandwidthTest(int device_id1, int device_id2, ncclComm_t *comms, ncclUniqueId id) {
    // Memory allocation
    float* sendBuffer;
    float* recvBuffer;

    cudaStream_t s1 = (cudaStream_t)malloc(sizeof(cudaStream_t));
    cudaStream_t s2 = (cudaStream_t)malloc(sizeof(cudaStream_t));

    // Allocate buffer and prepare stream for each gpu
    cudaSetDevice(device_id1);
    cudaMalloc(&sendBuffer, kDataSize * sizeof(float));
    cudaMemset(sendBuffer, 1.0, kDataSize * sizeof(float));
    cudaStreamCreate(&s1);

    cudaSetDevice(device_id2);
    cudaMalloc(&recvBuffer, kDataSize * sizeof(float));
    cudaMemset(recvBuffer, 0.0, kDataSize * sizeof(float));
    cudaStreamCreate(&s2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record elapsed time for rank n send and rank m recv
    cudaEventRecord(start);
    ncclGroupStart();
    NCCL_CHECK(ncclSend(sendBuffer, kDataSize, ncclFloat, device_id2, comms[device_id1], s1));
    NCCL_CHECK(ncclRecv(recvBuffer, kDataSize, ncclFloat, device_id1, comms[device_id2], s2));
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    ncclGroupEnd();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
 
    // Clean up
    cudaSetDevice(device_id1);
    NCCL_CHECK(ncclCommDestroy(comms[device_id1]));
    cudaSetDevice(device_id2);
    NCCL_CHECK(ncclCommDestroy(comms[device_id2]));

    cudaSetDevice(device_id1);
    cudaFree(sendBuffer);

    cudaSetDevice(device_id2);
    cudaFree(recvBuffer);

    return milliseconds / 1000.0; // Convert to seconds
    return 0.0;
}

int main() {
    int version;
    ncclGetVersion(&version);
    printf("nccl version: %d\n", version);

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    printf("device count: %d\n", numGPUs);

    if (numGPUs < 2) {
        std::cout << "Error: At least two GPUs are required." << std::endl;
        return 0;
    }

    ncclComm_t comms[numGPUs];
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    // create communicator for each GPU in a single node
    ncclCommInitAll(comms, numGPUs, NULL);
    // for(int i = 0; i < numGPUs; i++) {
    //     cudaSetDevice(i);
    //     ncclCommInitRank(&(comms[i]), numGPUs, id, i);
    // }

    for (int i = 0; i < numGPUs; i++) {
        for (int j = 0; j < numGPUs; j++) {
            float time = P2PBandwidthTest(i, j, comms, id);
            float bandwidth = (kDataSize * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / time; // in GB/s
            printf("GPU %d -> GPU %d: Bandwidth: %10.2f GB/s\n", i, j, bandwidth);
        }
    }

    return 0;
}