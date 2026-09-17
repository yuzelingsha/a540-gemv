// AdrenoLLM INT4 Block-32 Decoding Kernel for Adreno 540
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void int4_block32_gemv(
    __global const uchar* weights,      // INT4 packed weights (2 elements per byte)
    __global const half* scales,        // Block scales (1 scale per 32 weights)
    __global const half* input_vec,     // FP16 input activation vector
    __global half* output_vec,          // FP16 output vector
    const int K,                        // Input dimension
    const int N                         // Output dimension
) {
    int row = get_global_id(0);
    if (row >= N) return;

    half acc = 0.0h;
    int blocks_per_row = K / 32;

    for (int b = 0; b < blocks_per_row; ++b) {
        half s = scales[row * blocks_per_row + b];
        int w_offset = (row * (K / 2)) + (b * 16);
        int in_offset = b * 32;

        #pragma unroll 4
        for (int i = 0; i < 16; ++i) {
            uchar packed = weights[w_offset + i];
            int q0 = (int)(packed & 0x0F) - 8;
            int q1 = (int)((packed >> 4) & 0x0F) - 8;

            half w0 = (half)q0 * s;
            half w1 = (half)q1 * s;

            acc += w0 * input_vec[in_offset + 2 * i];
            acc += w1 * input_vec[in_offset + 2 * i + 1];
        }
    }

    output_vec[row] = acc;
}
