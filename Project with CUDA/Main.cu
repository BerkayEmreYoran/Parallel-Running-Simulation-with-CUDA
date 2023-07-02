#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Runner.cuh"

void delay(int milliseconds) {
    clock_t start_time = clock();
    while (clock() < start_time + milliseconds);
}

__global__ void updatePositions(Runner* runners) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    runners[tid].updatePosition();
}

int main() {
    Runner runners[NUM_RUNNERS];
    int numBlocks = 1;
    int threadsPerBlock = NUM_RUNNERS;
    int step = 1;
    int updateInterval = 1000 / TIME_INTERVAL; // Konum güncelleme aralýðý

    // Koþucularýn konumunu ve hýzýný bellekte CUDA global hafýzasýna kopyalama
    Runner* d_runners;
    cudaMalloc((void**)&d_runners, NUM_RUNNERS * sizeof(Runner));
    cudaMemcpy(d_runners, runners, NUM_RUNNERS * sizeof(Runner), cudaMemcpyHostToDevice);

    while (1) {
        // Koþucularýn konumunu güncelleme
        dim3 numBlocks(1);
        dim3 threadsPerBlock(NUM_RUNNERS);
        updatePositions << <numBlocks, threadsPerBlock >> > (d_runners);
        cudaDeviceSynchronize();

        // Koþucularýn konumunu ve hýzýný CUDA global hafýzasýndan ana belleðe kopyalama
        cudaMemcpy(runners, d_runners, NUM_RUNNERS * sizeof(Runner), cudaMemcpyDeviceToHost);

        // Bitiþ çizgisine ilk ulaþan koþucunun indeksini bulma
        int winnerIndex = -1;
        for (int j = 0; j < NUM_RUNNERS; j++) {
            if (runners[j].position >= RACE_DISTANCE) {
                winnerIndex = j;
                break;
            }
        }

        // Bitiþ çizgisine ulaþan tüm koþucularýn konumunu yazdýrma
        if (winnerIndex != -1) {
            printf("Bitis cizgisine ulasan kosucularin konumu:\n");
            for (int j = 0; j < NUM_RUNNERS; j++) {
                printf("Kosucu %d: %.2f metre\n", j + 1, runners[j].position);
            }
            break;
        }

        if (step % updateInterval == 0) {
            printf("Kosucularin anlik konumu (Saniye: %d):\n", step);
            for (int j = 0; j < NUM_RUNNERS; j++) {
                printf("Kosucu %d: %.2f metre\n", j + 1, runners[j].position);
            }
            printf("\n");
        }

        delay(TIME_INTERVAL);
        step++;
    }

    // Yarýþýn sýralamasýný hesaplama
    int sortedIndices[NUM_RUNNERS];
    for (int i = 0; i < NUM_RUNNERS; i++) {
        sortedIndices[i] = i;
    }

    // Sýralama iþlemi
    for (int i = 0; i < NUM_RUNNERS - 1; i++) {
        for (int j = i + 1; j < NUM_RUNNERS; j++) {
            if (runners[sortedIndices[i]].position < runners[sortedIndices[j]].position) {
                int temp = sortedIndices[i];
                sortedIndices[i] = sortedIndices[j];
                sortedIndices[j] = temp;
            }
        }
    }

    // Yarýþýn sýralamasýný yazdýrma
    printf("Yarisinn siralamasi:\n");
    for (int i = 0; i < NUM_RUNNERS; i++) {
        printf("Sira %d: Kosucu %d\n", i + 1, sortedIndices[i] + 1);
    }

    // Bellek temizleme
    cudaFree(d_runners);

    return 0;
}