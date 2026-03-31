#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

struct COOMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;

    COOMatrix()
        : numRows(0), numCols(0), numNonzeros(0),
          rowIdxs(nullptr), colIdxs(nullptr), values(nullptr) {}

    COOMatrix(unsigned int rows, unsigned int cols, unsigned int nnz)
        : numRows(rows), numCols(cols), numNonzeros(nnz),
          rowIdxs(nullptr), colIdxs(nullptr), values(nullptr) {
        if (nnz > 0) {
            rowIdxs = new unsigned int[nnz];
            colIdxs = new unsigned int[nnz];
            values  = new float[nnz];
        }
    }

    ~COOMatrix() {
        delete[] rowIdxs;
        delete[] colIdxs;
        delete[] values;
    }

    COOMatrix(const COOMatrix&) = delete;
    COOMatrix& operator=(const COOMatrix&) = delete;

    COOMatrix(COOMatrix&& other) noexcept
        : numRows(other.numRows),
          numCols(other.numCols),
          numNonzeros(other.numNonzeros),
          rowIdxs(other.rowIdxs),
          colIdxs(other.colIdxs),
          values(other.values) {
        other.numRows = 0;
        other.numCols = 0;
        other.numNonzeros = 0;
        other.rowIdxs = nullptr;
        other.colIdxs = nullptr;
        other.values = nullptr;
    }

    COOMatrix& operator=(COOMatrix&& other) noexcept {
        if (this != &other) {
            delete[] rowIdxs;
            delete[] colIdxs;
            delete[] values;

            numRows = other.numRows;
            numCols = other.numCols;
            numNonzeros = other.numNonzeros;
            rowIdxs = other.rowIdxs;
            colIdxs = other.colIdxs;
            values = other.values;

            other.numRows = 0;
            other.numCols = 0;
            other.numNonzeros = 0;
            other.rowIdxs = nullptr;
            other.colIdxs = nullptr;
            other.values = nullptr;
        }
        return *this;
    }
};

struct COOMatrixDevice {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    const unsigned int* rowIdxs;
    const unsigned int* colIdxs;
    const float* values;
};

struct GpuTiming {
    float h2dMs = 0.0f;
    float kernelMs = 0.0f;
    float d2hMs = 0.0f;
    float totalMs = 0.0f;
};

void generateDenseMatrix(std::vector<float>& denseMatrix,
                         unsigned int rows,
                         unsigned int cols,
                         float sparsity,
                         unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> valDist(-1.0f, 1.0f);

    denseMatrix.resize(rows * cols);

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            if (prob(rng) < sparsity) {
                float v = valDist(rng);
                if (std::fabs(v) < 1e-6f) v = 0.5f;
                denseMatrix[i * cols + j] = v;
            } else {
                denseMatrix[i * cols + j] = 0.0f;
            }
        }
    }
}

void generateVector(std::vector<float>& vec, unsigned int n, unsigned int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    vec.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
        vec[i] = dist(rng);
    }
}

COOMatrix denseToCOO(const std::vector<float>& denseMatrix, unsigned int rows, unsigned int cols) {
    unsigned int nnz = 0;
    for (unsigned int i = 0; i < rows * cols; ++i) {
        if (denseMatrix[i] != 0.0f) {
            ++nnz;
        }
    }

    COOMatrix coo(rows, cols, nnz);
    unsigned int idx = 0;

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            float v = denseMatrix[i * cols + j];
            if (v != 0.0f) {
                coo.rowIdxs[idx] = i;
                coo.colIdxs[idx] = j;
                coo.values[idx] = v;
                ++idx;
            }
        }
    }

    return coo;
}

void spmv_coo_cpu(const COOMatrix& coo, const float* inVector, float* outVector) {
    std::fill(outVector, outVector + coo.numRows, 0.0f);

    for (unsigned int i = 0; i < coo.numNonzeros; ++i) {
        unsigned int row = coo.rowIdxs[i];
        unsigned int col = coo.colIdxs[i];
        outVector[row] += coo.values[i] * inVector[col];
    }
}

__global__ void spmv_coo_kernel(COOMatrixDevice coo, const float* inVector, float* outVector) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < coo.numNonzeros) {
        unsigned int row = coo.rowIdxs[i];
        unsigned int col = coo.colIdxs[i];
        float value = coo.values[i];
        atomicAdd(&outVector[row], value * inVector[col]);
    }
}

float maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

float spmv_coo_gpu(const COOMatrix& cooMatrix,
                   const float* h_inVector,
                   float* h_outVector,
                   GpuTiming* timing = nullptr) {
    COOMatrixDevice d_coo{};
    d_coo.numRows = cooMatrix.numRows;
    d_coo.numCols = cooMatrix.numCols;
    d_coo.numNonzeros = cooMatrix.numNonzeros;

    unsigned int* d_rowIdxs = nullptr;
    unsigned int* d_colIdxs = nullptr;
    float* d_values = nullptr;
    float* d_inVector = nullptr;
    float* d_outVector = nullptr;

    cudaMalloc((void**)&d_rowIdxs, d_coo.numNonzeros * sizeof(unsigned int));
    cudaMalloc((void**)&d_colIdxs, d_coo.numNonzeros * sizeof(unsigned int));
    cudaMalloc((void**)&d_values,  d_coo.numNonzeros * sizeof(float));
    cudaMalloc((void**)&d_inVector, d_coo.numCols * sizeof(float));
    cudaMalloc((void**)&d_outVector, d_coo.numRows * sizeof(float));

    d_coo.rowIdxs = d_rowIdxs;
    d_coo.colIdxs = d_colIdxs;
    d_coo.values  = d_values;

    cudaEvent_t startTotal, stopTotal;
    cudaEvent_t startH2D, stopH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startD2H, stopD2H;

    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    cudaEventRecord(startTotal);

    cudaEventRecord(startH2D);
    cudaMemcpy(d_rowIdxs, cooMatrix.rowIdxs,
               d_coo.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxs, cooMatrix.colIdxs,
               d_coo.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, cooMatrix.values,
               d_coo.numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVector, h_inVector,
               d_coo.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_outVector, 0, d_coo.numRows * sizeof(float));
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);

    unsigned int threadsPerBlock = 256;
    unsigned int numBlocks = (d_coo.numNonzeros + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(startKernel);
    spmv_coo_kernel<<<numBlocks, threadsPerBlock>>>(d_coo, d_inVector, d_outVector);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_outVector, d_outVector,
               d_coo.numRows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);

    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);

    float h2dMs = 0.0f;
    float kernelMs = 0.0f;
    float d2hMs = 0.0f;
    float totalMs = 0.0f;

    cudaEventElapsedTime(&h2dMs, startH2D, stopH2D);
    cudaEventElapsedTime(&kernelMs, startKernel, stopKernel);
    cudaEventElapsedTime(&d2hMs, startD2H, stopD2H);
    cudaEventElapsedTime(&totalMs, startTotal, stopTotal);

    if (timing) {
        timing->h2dMs = h2dMs;
        timing->kernelMs = kernelMs;
        timing->d2hMs = d2hMs;
        timing->totalMs = totalMs;
    }

    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);

    cudaFree(d_rowIdxs);
    cudaFree(d_colIdxs);
    cudaFree(d_values);
    cudaFree(d_inVector);
    cudaFree(d_outVector);

    return kernelMs;
}

int main(int argc, char** argv) {
    unsigned int rows = 4096;
    unsigned int cols = 4096;
    float sparsity = 0.01f;

    if (argc >= 2) rows = static_cast<unsigned int>(std::atoi(argv[1]));
    if (argc >= 3) cols = static_cast<unsigned int>(std::atoi(argv[2]));
    if (argc >= 4) sparsity = std::atof(argv[3]);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "rows = " << rows
              << ", cols = " << cols
              << ", sparsity = " << sparsity << std::endl;

    std::vector<float> denseMatrix;
    generateDenseMatrix(denseMatrix, rows, cols, sparsity);

    COOMatrix coo = denseToCOO(denseMatrix, rows, cols);
    std::cout << "nnz = " << coo.numNonzeros << std::endl;

    std::vector<float> inVector;
    generateVector(inVector, cols);

    std::vector<float> outCpu(rows, 0.0f);
    std::vector<float> outGpu(rows, 0.0f);

    spmv_coo_cpu(coo, inVector.data(), outCpu.data());

    GpuTiming timing;
    spmv_coo_gpu(coo, inVector.data(), outGpu.data(), &timing);

    float err = maxAbsError(outCpu, outGpu);

    std::cout << "H2D time   = " << timing.h2dMs << " ms" << std::endl;
    std::cout << "Kernel time= " << timing.kernelMs << " ms" << std::endl;
    std::cout << "D2H time   = " << timing.d2hMs << " ms" << std::endl;
    std::cout << "Total time = " << timing.totalMs << " ms" << std::endl;
    std::cout << "Max abs error = " << err << std::endl;

    std::cout << "First 10 output elements:" << std::endl;
    for (int i = 0; i < 10 && i < static_cast<int>(rows); ++i) {
        std::cout << "y[" << i << "] cpu = " << outCpu[i]
                  << ", gpu = " << outGpu[i] << std::endl;
    }

    if (err < 1e-4f) {
        std::cout << "Result verified." << std::endl;
    } else {
        std::cout << "Result mismatch!" << std::endl;
    }

    return 0;
}