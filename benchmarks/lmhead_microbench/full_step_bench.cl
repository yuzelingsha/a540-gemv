#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// LM-Head with Workgroup LMS staging (Opt 2)
__kernel void lmhead_opt2_lms(
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
