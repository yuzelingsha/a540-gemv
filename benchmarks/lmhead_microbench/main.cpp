#define CL_TARGET_OPENCL_VERSION 200
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <dlfcn.h>
#include <CL/cl.h>

static void* g_cl_handle = nullptr;
#define DECL(fn) static decltype(&fn) p_##fn = nullptr;
DECL(clGetPlatformIDs);
DECL(clGetDeviceIDs);
DECL(clGetDeviceInfo);
DECL(clCreateContext);
DECL(clCreateCommandQueue);
DECL(clCreateProgramWithSource);
DECL(clBuildProgram);
DECL(clGetProgramBuildInfo);
DECL(clCreateBuffer);
DECL(clEnqueueWriteBuffer);
DECL(clCreateKernel);
DECL(clSetKernelArg);
DECL(clEnqueueNDRangeKernel);
DECL(clWaitForEvents);
DECL(clGetEventProfilingInfo);
DECL(clReleaseEvent);
DECL(clReleaseKernel);
DECL(clReleaseMemObject);
DECL(clReleaseProgram);
DECL(clReleaseCommandQueue);
DECL(clReleaseContext);
DECL(clFinish);

void load_opencl() {
    const char* paths[] = {
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "libOpenCL.so"
    };
    for (const char* path : paths) {
        g_cl_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (g_cl_handle) break;
    }
    if (!g_cl_handle) {
        fprintf(stderr, "Failed to dlopen libOpenCL.so: %s\n", dlerror());
        exit(1);
    }
#define LOAD(fn) p_##fn = (decltype(&fn))dlsym(g_cl_handle, #fn); if (!p_##fn) { fprintf(stderr, "Missing " #fn "\n"); exit(1); }
    LOAD(clGetPlatformIDs);
    LOAD(clGetDeviceIDs);
    LOAD(clGetDeviceInfo);
    LOAD(clCreateContext);
    LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram);
    LOAD(clGetProgramBuildInfo);
    LOAD(clCreateBuffer);
    LOAD(clEnqueueWriteBuffer);
    LOAD(clCreateKernel);
    LOAD(clSetKernelArg);
    LOAD(clEnqueueNDRangeKernel);
    LOAD(clWaitForEvents);
    LOAD(clGetEventProfilingInfo);
    LOAD(clReleaseEvent);
    LOAD(clReleaseKernel);
    LOAD(clReleaseMemObject);
    LOAD(clReleaseProgram);
    LOAD(clReleaseCommandQueue);
    LOAD(clReleaseContext);
    LOAD(clFinish);
}

#define CHECK_CL(call) do { \
    cl_int err = (p_##call); \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

std::string load_file(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

struct PerfStats {
    double min_ms;
    double median_ms;
    double p95_ms;
    double max_ms;
    double bandwidth_gb_s;
};

PerfStats measure_kernel(cl_command_queue queue, cl_kernel kernel, size_t global_size, size_t local_size, size_t bytes_transferred, int warmups = 10, int iters = 50) {
    for (int i = 0; i < warmups; ++i) {
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, local_size > 0 ? &local_size : nullptr, 0, nullptr, nullptr));
    }
    CHECK_CL(clFinish(queue));

    std::vector<double> times_ms;
    times_ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        cl_event ev;
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, local_size > 0 ? &local_size : nullptr, 0, nullptr, &ev));
        CHECK_CL(clWaitForEvents(1, &ev));

        cl_ulong t_start = 0, t_end = 0;
        CHECK_CL(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr));
        CHECK_CL(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr));
        CHECK_CL(clReleaseEvent(ev));

        double dur_ms = (double)(t_end - t_start) * 1e-6;
        times_ms.push_back(dur_ms);
    }

    std::sort(times_ms.begin(), times_ms.end());
    double min_ms = times_ms.front();
    double max_ms = times_ms.back();
    double median_ms = times_ms[times_ms.size() / 2];
    double p95_ms = times_ms[(size_t)(times_ms.size() * 0.95)];
    double bw_gb_s = (double)bytes_transferred / (median_ms * 1e-3) / 1e9;

    return {min_ms, median_ms, p95_ms, max_ms, bw_gb_s};
}

int main(int argc, char** argv) {
    load_opencl();

    const int VOCAB_SIZE = 130560;
    const int HIDDEN_DIM = 1536;
    const size_t WEIGHTS_BYTES = (size_t)VOCAB_SIZE * HIDDEN_DIM;
    const size_t SCALES_BYTES = (size_t)VOCAB_SIZE * 2;
    const size_t HIDDEN_BYTES = (size_t)HIDDEN_DIM * 2;
    const size_t LOGITS_BYTES = (size_t)VOCAB_SIZE * 2;
    const int NUM_UINT4_PER_ROW = HIDDEN_DIM / 16;

    printf("=== Adreno 540 LM-Head Microbenchmark (P0 Suite) ===\n");
    printf("Matrix Shape: [%d, %d] INT8 | Weights Size: %.2f MB\n", VOCAB_SIZE, HIDDEN_DIM, (double)WEIGHTS_BYTES / (1024.0 * 1024.0));

    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms = 0, num_devices = 0;
    CHECK_CL(clGetPlatformIDs(1, &platform, &num_platforms));
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices));

    char dev_name[256];
    CHECK_CL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr));
    printf("OpenCL Device: %s\n", dev_name);

    cl_int err;
    cl_context ctx = p_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateContext failed: %d\n", err); exit(1); }

    cl_command_queue queue = p_clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue failed: %d\n", err); exit(1); }

    std::string cl_path = (argc > 1) ? argv[1] : "/data/local/tmp/kernels.cl";
    std::string source = load_file(cl_path.c_str());
    const char* src_ptr = source.c_str();
    size_t src_len = source.length();

    cl_program prog = p_clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource failed: %d\n", err); exit(1); }

    const char* build_opts = "-cl-std=CL2.0";
    err = p_clBuildProgram(prog, 1, &device, build_opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        p_clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);
        std::vector<char> log(log_len + 1);
        p_clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), nullptr);
        fprintf(stderr, "Build error:\n%s\n", log.data());
        exit(1);
    }

    cl_mem d_weights = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, WEIGHTS_BYTES, nullptr, &err);
    cl_mem d_hidden = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, HIDDEN_BYTES, nullptr, &err);
    cl_mem d_scales = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, SCALES_BYTES, nullptr, &err);
    cl_mem d_logits = p_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, LOGITS_BYTES, nullptr, &err);
    cl_mem d_dummy = p_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)VOCAB_SIZE * sizeof(float), nullptr, &err);

    std::vector<uint8_t> h_weights(WEIGHTS_BYTES, 1);
    std::vector<uint16_t> h_scales(VOCAB_SIZE, 0x3c00);
    std::vector<uint16_t> h_hidden(HIDDEN_DIM, 0x3c00);

    CHECK_CL(clEnqueueWriteBuffer(queue, d_weights, CL_TRUE, 0, WEIGHTS_BYTES, h_weights.data(), 0, nullptr, nullptr));
    CHECK_CL(clEnqueueWriteBuffer(queue, d_scales, CL_TRUE, 0, SCALES_BYTES, h_scales.data(), 0, nullptr, nullptr));
    CHECK_CL(clEnqueueWriteBuffer(queue, d_hidden, CL_TRUE, 0, HIDDEN_BYTES, h_hidden.data(), 0, nullptr, nullptr));

    size_t global_size = VOCAB_SIZE;
    size_t local_size = 64;

    const char* kernel_names[] = {
        "stage_a_weight_streaming",
        "stage_b_weight_unpack",
        "stage_c_weight_dequant",
        "stage_d_dot_product",
        "stage_d_constant_3072",
        "stage_d_local_stage",
        "stage_e_full_lmhead",
        "stage_e_lms_full_lmhead"
    };
    const char* stage_desc[] = {
        "Stage A (Pure Weight Streaming)",
        "Stage B (Weight + Unpack)",
        "Stage C (Weight + Dequant)",
        "Stage D (Global Dot-Product)",
        "Opt 1 (3072B Constant Memory)",
        "Opt 2 (LMS Workgroup Staging)",
        "Stage E (Full LM-Head Pipeline)",
        "Stage E Opt 2 (Full LMS Pipeline)"
    };

    printf("\n--- Profiling Stages (50 runs each) ---\n");
    for (int i = 0; i < 8; ++i) {
        cl_kernel k = p_clCreateKernel(prog, kernel_names[i], &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel %s failed: %d\n", kernel_names[i], err); exit(1); }

        CHECK_CL(clSetKernelArg(k, 0, sizeof(cl_mem), &d_weights));
        if (i == 0 || i == 1) {
            CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_dummy));
            CHECK_CL(clSetKernelArg(k, 2, sizeof(int), &NUM_UINT4_PER_ROW));
        } else if (i == 2) {
            CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_scales));
            CHECK_CL(clSetKernelArg(k, 2, sizeof(cl_mem), &d_dummy));
            CHECK_CL(clSetKernelArg(k, 3, sizeof(int), &NUM_UINT4_PER_ROW));
        } else if (i == 3 || i == 4 || i == 5) {
            CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_hidden));
            CHECK_CL(clSetKernelArg(k, 2, sizeof(cl_mem), &d_scales));
            CHECK_CL(clSetKernelArg(k, 3, sizeof(cl_mem), &d_dummy));
            CHECK_CL(clSetKernelArg(k, 4, sizeof(int), &NUM_UINT4_PER_ROW));
        } else if (i == 6 || i == 7) {
            CHECK_CL(clSetKernelArg(k, 1, sizeof(cl_mem), &d_hidden));
            CHECK_CL(clSetKernelArg(k, 2, sizeof(cl_mem), &d_scales));
            CHECK_CL(clSetKernelArg(k, 3, sizeof(cl_mem), &d_logits));
            CHECK_CL(clSetKernelArg(k, 4, sizeof(int), &NUM_UINT4_PER_ROW));
        }

        size_t num_workgroups = (global_size + local_size - 1) / local_size;
        size_t total_transferred = WEIGHTS_BYTES;
        if (i == 0 || i == 1) {
            total_transferred = WEIGHTS_BYTES;
        } else if (i == 2) {
            total_transferred = WEIGHTS_BYTES + SCALES_BYTES;
        } else if (i == 3) {
            // Uncached global memory dot-product: each work-item reads 3072B hidden vector
            total_transferred = WEIGHTS_BYTES + SCALES_BYTES + (size_t)VOCAB_SIZE * HIDDEN_BYTES;
        } else if (i == 4) {
            // Constant memory: broadcast/cached on chip
            total_transferred = WEIGHTS_BYTES + SCALES_BYTES + HIDDEN_BYTES;
        } else if (i == 5) {
            // LMS staging: each workgroup cooperatively loads 3072B hidden vector once
            total_transferred = WEIGHTS_BYTES + SCALES_BYTES + num_workgroups * HIDDEN_BYTES;
        } else if (i == 6) {
            // Full LM-head unoptimized: global hidden vector read by each work-item + logits output
            total_transferred = WEIGHTS_BYTES + SCALES_BYTES + (size_t)VOCAB_SIZE * HIDDEN_BYTES + LOGITS_BYTES;
        } else if (i == 7) {
            // Full LM-head LMS: cooperative hidden vector preload per workgroup + logits output
            total_transferred = WEIGHTS_BYTES + SCALES_BYTES + num_workgroups * HIDDEN_BYTES + LOGITS_BYTES;
        }

        PerfStats stats = measure_kernel(queue, k, global_size, local_size, total_transferred, 10, 50);
        printf("%-32s : Median: %6.2f ms | P95: %6.2f ms | Min: %6.2f ms | Effective BW: %5.2f GB/s\n",
               stage_desc[i], stats.median_ms, stats.p95_ms, stats.min_ms, stats.bandwidth_gb_s);

        p_clReleaseKernel(k);
    }

    printf("\n=== Microbenchmark Complete ===\n");

    p_clReleaseMemObject(d_weights);
    p_clReleaseMemObject(d_hidden);
    p_clReleaseMemObject(d_scales);
    p_clReleaseMemObject(d_logits);
    p_clReleaseMemObject(d_dummy);
    p_clReleaseProgram(prog);
    p_clReleaseCommandQueue(queue);
    p_clReleaseContext(ctx);

    return 0;
}
