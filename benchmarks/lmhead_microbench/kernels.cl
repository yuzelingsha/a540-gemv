// OpenCL microbenchmarks for Adreno 540 LM-Head (130560 x 1536 INT8)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Stage A: Pure weight streaming (read weight buffer as uchar4/uint4)
__kernel void stage_a_weight_streaming(
    __global const uint4* restrict weights,
    __global float* restrict dummy_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    uint acc = 0;
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        acc ^= (w.x ^ w.y ^ w.z ^ w.w);
    }
    dummy_out[row] = (float)acc;
}

// Stage B: Weight streaming + unpack to signed chars
__kernel void stage_b_weight_unpack(
    __global const uint4* restrict weights,
    __global float* restrict dummy_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    int acc = 0;
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        // Unpack 16 bytes into signed integers (sign-extended)
        char4 b0 = as_char4(w.x);
        char4 b1 = as_char4(w.y);
        char4 b2 = as_char4(w.z);
        char4 b3 = as_char4(w.w);
        acc += (int)b0.x + (int)b0.y + (int)b0.z + (int)b0.w;
        acc += (int)b1.x + (int)b1.y + (int)b1.z + (int)b1.w;
        acc += (int)b2.x + (int)b2.y + (int)b2.z + (int)b2.w;
        acc += (int)b3.x + (int)b3.y + (int)b3.z + (int)b3.w;
    }
    dummy_out[row] = (float)acc;
}

// Stage C: Weight unpack + dequant (multiply scale)
__kernel void stage_c_weight_dequant(
    __global const uint4* restrict weights,
    __global const half* restrict scales,
    __global float* restrict dummy_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    half scale = scales[row];
    half acc = 0.0h;
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        char4 b0 = as_char4(w.x);
        char4 b1 = as_char4(w.y);
        char4 b2 = as_char4(w.z);
        char4 b3 = as_char4(w.w);
        acc += ((half)b0.x + (half)b0.y + (half)b0.z + (half)b0.w) * scale;
        acc += ((half)b1.x + (half)b1.y + (half)b1.z + (half)b1.w) * scale;
        acc += ((half)b2.x + (half)b2.y + (half)b2.z + (half)b2.w) * scale;
        acc += ((half)b3.x + (half)b3.y + (half)b3.z + (half)b3.w) * scale;
    }
    dummy_out[row] = (float)acc;
}

// Stage D: Full dot-product with 1536 hidden vector
__kernel void stage_d_dot_product(
    __global const uint4* restrict weights,
    __global const half4* restrict hidden_vec,
    __global const half* restrict scales,
    __global float* restrict dummy_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    half scale = scales[row];
    float acc = 0.0f;
    
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        half4 x0 = hidden_vec[k * 4 + 0];
        half4 x1 = hidden_vec[k * 4 + 1];
        half4 x2 = hidden_vec[k * 4 + 2];
        half4 x3 = hidden_vec[k * 4 + 3];
        
        uchar4 b0 = as_uchar4(w.x);
        uchar4 b1 = as_uchar4(w.y);
        uchar4 b2 = as_uchar4(w.z);
        uchar4 b3 = as_uchar4(w.w);
        
        acc += (float)((half)((char)b0.x) * x0.x);
        acc += (float)((half)((char)b0.y) * x0.y);
        acc += (float)((half)((char)b0.z) * x0.z);
        acc += (float)((half)((char)b0.w) * x0.w);
        
        acc += (float)((half)((char)b1.x) * x1.x);
        acc += (float)((half)((char)b1.y) * x1.y);
        acc += (float)((half)((char)b1.z) * x1.z);
        acc += (float)((half)((char)b1.w) * x1.w);
        
        acc += (float)((half)((char)b2.x) * x2.x);
        acc += (float)((half)((char)b2.y) * x2.y);
        acc += (float)((half)((char)b2.z) * x2.z);
        acc += (float)((half)((char)b2.w) * x2.w);
        
        acc += (float)((half)((char)b3.x) * x3.x);
        acc += (float)((half)((char)b3.y) * x3.y);
        acc += (float)((half)((char)b3.z) * x3.z);
        acc += (float)((half)((char)b3.w) * x3.w);
    }
    acc *= (float)scale;
    dummy_out[row] = acc;
}

// Stage E: Full LM-Head with logits writeback
__kernel void stage_e_full_lmhead(
    __global const uint4* restrict weights,
    __global const half4* restrict hidden_vec,
    __global const half* restrict scales,
    __global half* restrict logits_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    half scale = scales[row];
    float acc = 0.0f;
    
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        half4 x0 = hidden_vec[k * 4 + 0];
        half4 x1 = hidden_vec[k * 4 + 1];
        half4 x2 = hidden_vec[k * 4 + 2];
        half4 x3 = hidden_vec[k * 4 + 3];
        
        uchar4 b0 = as_uchar4(w.x);
        uchar4 b1 = as_uchar4(w.y);
        uchar4 b2 = as_uchar4(w.z);
        uchar4 b3 = as_uchar4(w.w);
        
        acc += (float)((half)((char)b0.x) * x0.x);
        acc += (float)((half)((char)b0.y) * x0.y);
        acc += (float)((half)((char)b0.z) * x0.z);
        acc += (float)((half)((char)b0.w) * x0.w);
        
        acc += (float)((half)((char)b1.x) * x1.x);
        acc += (float)((half)((char)b1.y) * x1.y);
        acc += (float)((half)((char)b1.z) * x1.z);
        acc += (float)((half)((char)b1.w) * x1.w);
        
        acc += (float)((half)((char)b2.x) * x2.x);
        acc += (float)((half)((char)b2.y) * x2.y);
        acc += (float)((half)((char)b2.z) * x2.z);
        acc += (float)((half)((char)b2.w) * x2.w);
        
        acc += (float)((half)((char)b3.x) * x3.x);
        acc += (float)((half)((char)b3.y) * x3.y);
        acc += (float)((half)((char)b3.z) * x3.z);
        acc += (float)((half)((char)b3.w) * x3.w);
    }
    logits_out[row] = (half)(acc * (float)scale);
}

// Optimization 1: Full 3072B Hidden Vector in On-Chip Constant Memory
__kernel void stage_d_constant_3072(
    __global const uint4* restrict weights,
    __constant const half4* restrict hidden_vec,
    __global const half* restrict scales,
    __global float* restrict dummy_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    half scale = scales[row];
    float acc = 0.0f;
    
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        half4 x0 = hidden_vec[k * 4 + 0];
        half4 x1 = hidden_vec[k * 4 + 1];
        half4 x2 = hidden_vec[k * 4 + 2];
        half4 x3 = hidden_vec[k * 4 + 3];
        
        uchar4 b0 = as_uchar4(w.x);
        uchar4 b1 = as_uchar4(w.y);
        uchar4 b2 = as_uchar4(w.z);
        uchar4 b3 = as_uchar4(w.w);
        
        acc += (float)((half)((char)b0.x) * x0.x);
        acc += (float)((half)((char)b0.y) * x0.y);
        acc += (float)((half)((char)b0.z) * x0.z);
        acc += (float)((half)((char)b0.w) * x0.w);
        
        acc += (float)((half)((char)b1.x) * x1.x);
        acc += (float)((half)((char)b1.y) * x1.y);
        acc += (float)((half)((char)b1.z) * x1.z);
        acc += (float)((half)((char)b1.w) * x1.w);
        
        acc += (float)((half)((char)b2.x) * x2.x);
        acc += (float)((half)((char)b2.y) * x2.y);
        acc += (float)((half)((char)b2.z) * x2.z);
        acc += (float)((half)((char)b2.w) * x2.w);
        
        acc += (float)((half)((char)b3.x) * x3.x);
        acc += (float)((half)((char)b3.y) * x3.y);
        acc += (float)((half)((char)b3.z) * x3.z);
        acc += (float)((half)((char)b3.w) * x3.w);
    }
    acc *= (float)scale;
    dummy_out[row] = acc;
}

// Optimization 2: Workgroup Local Memory Cooperative Staging (64 threads stage 1536 halves = 3072 bytes)
__kernel void stage_d_local_stage(
    __global const uint4* restrict weights,
    __global const half4* restrict global_hidden_vec,
    __global const half* restrict scales,
    __global float* restrict dummy_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    
    // 1536 halfs = 384 half4s
    __local half4 local_hidden[384];
    
    // Cooperatively load hidden vector to Local Memory Storage (LMS)
    for (int i = lid; i < 384; i += lsize) {
        local_hidden[i] = global_hidden_vec[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    half scale = scales[row];
    float acc = 0.0f;
    
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        half4 x0 = local_hidden[k * 4 + 0];
        half4 x1 = local_hidden[k * 4 + 1];
        half4 x2 = local_hidden[k * 4 + 2];
        half4 x3 = local_hidden[k * 4 + 3];
        
        uchar4 b0 = as_uchar4(w.x);
        uchar4 b1 = as_uchar4(w.y);
        uchar4 b2 = as_uchar4(w.z);
        uchar4 b3 = as_uchar4(w.w);
        
        acc += (float)((half)((char)b0.x) * x0.x);
        acc += (float)((half)((char)b0.y) * x0.y);
        acc += (float)((half)((char)b0.z) * x0.z);
        acc += (float)((half)((char)b0.w) * x0.w);
        
        acc += (float)((half)((char)b1.x) * x1.x);
        acc += (float)((half)((char)b1.y) * x1.y);
        acc += (float)((half)((char)b1.z) * x1.z);
        acc += (float)((half)((char)b1.w) * x1.w);
        
        acc += (float)((half)((char)b2.x) * x2.x);
        acc += (float)((half)((char)b2.y) * x2.y);
        acc += (float)((half)((char)b2.z) * x2.z);
        acc += (float)((half)((char)b2.w) * x2.w);
        
        acc += (float)((half)((char)b3.x) * x3.x);
        acc += (float)((half)((char)b3.y) * x3.y);
        acc += (float)((half)((char)b3.z) * x3.z);
        acc += (float)((half)((char)b3.w) * x3.w);
    }
    acc *= (float)scale;
    dummy_out[row] = acc;
}

// Full LM-Head Pipeline with Workgroup LMS Staging (Production Optimization)
__kernel void stage_e_lms_full_lmhead(
    __global const uint4* restrict weights,
    __global const half4* restrict global_hidden_vec,
    __global const half* restrict scales,
    __global half* restrict logits_out,
    const int num_uint4_per_row)
{
    int row = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    
    __local half4 local_hidden[384];
    for (int i = lid; i < 384; i += lsize) {
        local_hidden[i] = global_hidden_vec[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (row >= 130560) return;
    
    int row_offset = row * num_uint4_per_row;
    half scale = scales[row];
    float acc = 0.0f;
    
    for (int k = 0; k < num_uint4_per_row; ++k) {
        uint4 w = weights[row_offset + k];
        half4 x0 = local_hidden[k * 4 + 0];
        half4 x1 = local_hidden[k * 4 + 1];
        half4 x2 = local_hidden[k * 4 + 2];
        half4 x3 = local_hidden[k * 4 + 3];
        
        uchar4 b0 = as_uchar4(w.x);
        uchar4 b1 = as_uchar4(w.y);
        uchar4 b2 = as_uchar4(w.z);
        uchar4 b3 = as_uchar4(w.w);
        
        acc += (float)((half)((char)b0.x) * x0.x);
        acc += (float)((half)((char)b0.y) * x0.y);
        acc += (float)((half)((char)b0.z) * x0.z);
        acc += (float)((half)((char)b0.w) * x0.w);
        
        acc += (float)((half)((char)b1.x) * x1.x);
        acc += (float)((half)((char)b1.y) * x1.y);
        acc += (float)((half)((char)b1.z) * x1.z);
        acc += (float)((half)((char)b1.w) * x1.w);
        
        acc += (float)((half)((char)b2.x) * x2.x);
        acc += (float)((half)((char)b2.y) * x2.y);
        acc += (float)((half)((char)b2.z) * x2.z);
        acc += (float)((half)((char)b2.w) * x2.w);
        
        acc += (float)((half)((char)b3.x) * x3.x);
        acc += (float)((half)((char)b3.y) * x3.y);
        acc += (float)((half)((char)b3.z) * x3.z);
        acc += (float)((half)((char)b3.w) * x3.w);
    }
    logits_out[row] = (half)(acc * (float)scale);
}
