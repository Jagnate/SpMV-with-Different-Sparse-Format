#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

struct CSRMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;

    CSRMatrix()
        : numRows(0), numCols(0), numNonzeros(0),
          rowPtrs(nullptr), colIdxs(nullptr), values(nullptr) {}

    CSRMatrix(unsigned int rows, unsigned int cols, unsigned int nnz)
        : numRows(rows), numCols(cols), numNonzeros(nnz),
          rowPtrs(nullptr), colIdxs(nullptr), values(nullptr) {
        rowPtrs = new unsigned int[numRows + 1];
        if (numNonzeros > 0) {
            colIdxs = new unsigned int[numNonzeros];
            values = new float[numNonzeros];
        }
    }

    ~CSRMatrix() {
        delete[] rowPtrs;
        delete[] colIdxs;
        delete[] values;
    }

    CSRMatrix(const CSRMatrix&) = delete;
    CSRMatrix& operator=(const CSRMatrix&) = delete;

    CSRMatrix(CSRMatrix&& other) noexcept
        : numRows(other.numRows),
          numCols(other.numCols),
          numNonzeros(other.numNonzeros),
          rowPtrs(other.rowPtrs),
          colIdxs(other.colIdxs),
          values(other.values) {
        other.numRows = 0;
        other.numCols = 0;
        other.numNonzeros = 0;
        other.rowPtrs = nullptr;
        other.colIdxs = nullptr;
        other.values = nullptr;
    }

    CSRMatrix& operator=(CSRMatrix&& other) noexcept {
        if (this != &other) {
            delete[] rowPtrs;
            delete[] colIdxs;
            delete[] values;

            numRows = other.numRows;
            numCols = other.numCols;
            numNonzeros = other.numNonzeros;
            rowPtrs = other.rowPtrs;
            colIdxs = other.colIdxs;
            values = other.values;

            other.numRows = 0;
            other.numCols = 0;
            other.numNonzeros = 0;
            other.rowPtrs = nullptr;
            other.colIdxs = nullptr;
            other.values = nullptr;
        }
        return *this;
    }
};

struct CSRMatrixDevice {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    const unsigned int* rowPtrs;
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
                         float density,
                         unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> valDist(-1.0f, 1.0f);

    denseMatrix.resize(rows * cols);

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            if (prob(rng) < density) {
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

CSRMatrix denseToCSR(const std::vector<float>& denseMatrix,
                     unsigned int rows,
                     unsigned int cols) {
    unsigned int nnz = 0;
    for (unsigned int i = 0; i < rows * cols; ++i) {
        if (denseMatrix[i] != 0.0f) {
            ++nnz;
        }
    }

    CSRMatrix csr(rows, cols, nnz);

    unsigned int idx = 0;
    csr.rowPtrs[0] = 0;

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            float v = denseMatrix[i * cols + j];
            if (v != 0.0f) {
                csr.colIdxs[idx] = j;
                csr.values[idx] = v;
                ++idx;
            }
        }
        csr.rowPtrs[i + 1] = idx;
    }

    return csr;
}

void spmv_csr_cpu(const CSRMatrix& csr, const float* inVector, float* outVector) {
    for (unsigned int row = 0; row < csr.numRows; ++row) {
        float sum = 0.0f;
        for (unsigned int i = csr.rowPtrs[row]; i < csr.rowPtrs[row + 1]; ++i) {
            unsigned int col = csr.colIdxs[i];
            sum += csr.values[i] * inVector[col];
        }
        outVector[row] = sum;
    }
}

__global__ void spmv_csr_kernel(CSRMatrixDevice csr, const float* inVector, float* outVector) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < csr.numRows) {
        float sum = 0.0f;
        for (unsigned int i = csr.rowPtrs[row]; i < csr.rowPtrs[row + 1]; ++i) {
            unsigned int col = csr.colIdxs[i];
            float value = csr.values[i];
            sum += value * inVector[col];
        }
        outVector[row] = sum;
    }
}

float maxAbsError(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

float spmv_csr_gpu(const CSRMatrix& csrMatrix,
                   const float* h_inVector,
                   float* h_outVector,
                   GpuTiming* timing = nullptr) {
    CSRMatrixDevice d_csr{};
    d_csr.numRows = csrMatrix.numRows;
    d_csr.numCols = csrMatrix.numCols;
    d_csr.numNonzeros = csrMatrix.numNonzeros;

    unsigned int* d_rowPtrs = nullptr;
    unsigned int* d_colIdxs = nullptr;
    float* d_values = nullptr;
    float* d_inVector = nullptr;
    float* d_outVector = nullptr;

    cudaMalloc((void**)&d_rowPtrs, (d_csr.numRows + 1) * sizeof(unsigned int));
    cudaMalloc((void**)&d_colIdxs, d_csr.numNonzeros * sizeof(unsigned int));
    cudaMalloc((void**)&d_values,  d_csr.numNonzeros * sizeof(float));
    cudaMalloc((void**)&d_inVector, d_csr.numCols * sizeof(float));
    cudaMalloc((void**)&d_outVector, d_csr.numRows * sizeof(float));

    d_csr.rowPtrs = d_rowPtrs;
    d_csr.colIdxs = d_colIdxs;
    d_csr.values = d_values;

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
    cudaMemcpy(d_rowPtrs, csrMatrix.rowPtrs,
               (d_csr.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxs, csrMatrix.colIdxs,
               d_csr.numNonzeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, csrMatrix.values,
               d_csr.numNonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inVector, h_inVector,
               d_csr.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_outVector, 0, d_csr.numRows * sizeof(float));
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);

    unsigned int threadsPerBlock = 256;
    unsigned int numBlocks = (d_csr.numRows + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(startKernel);
    spmv_csr_kernel<<<numBlocks, threadsPerBlock>>>(d_csr, d_inVector, d_outVector);
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_outVector, d_outVector,
               d_csr.numRows * sizeof(float), cudaMemcpyDeviceToHost);
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

    cudaFree(d_rowPtrs);
    cudaFree(d_colIdxs);
    cudaFree(d_values);
    cudaFree(d_inVector);
    cudaFree(d_outVector);

    return kernelMs;
}

int main(int argc, char** argv) {
    unsigned int rows = 4096;
    unsigned int cols = 4096;
    float density = 0.01f;   // 这里表示非零比例，不是“0的比例”

    if (argc >= 2) rows = static_cast<unsigned int>(std::atoi(argv[1]));
    if (argc >= 3) cols = static_cast<unsigned int>(std::atoi(argv[2]));
    if (argc >= 4) density = std::atof(argv[3]);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "rows = " << rows
              << ", cols = " << cols
              << ", density = " << density << std::endl;

    std::vector<float> denseMatrix;
    generateDenseMatrix(denseMatrix, rows, cols, density);

    CSRMatrix csr = denseToCSR(denseMatrix, rows, cols);
    std::cout << "nnz = " << csr.numNonzeros << std::endl;

    std::vector<float> inVector;
    generateVector(inVector, cols);

    std::vector<float> outCpu(rows, 0.0f);
    std::vector<float> outGpu(rows, 0.0f);

    spmv_csr_cpu(csr, inVector.data(), outCpu.data());

    GpuTiming timing;
    spmv_csr_gpu(csr, inVector.data(), outGpu.data(), &timing);

    float err = maxAbsError(outCpu, outGpu);

    std::cout << "H2D time    = " << timing.h2dMs << " ms" << std::endl;
    std::cout << "Kernel time = " << timing.kernelMs << " ms" << std::endl;
    std::cout << "D2H time    = " << timing.d2hMs << " ms" << std::endl;
    std::cout << "Total time  = " << timing.totalMs << " ms" << std::endl;
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