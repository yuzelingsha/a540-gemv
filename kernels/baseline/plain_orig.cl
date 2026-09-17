#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  __read_only image2d_t weights_image2d,
  __read_only image2d_t weights_scale_image2d,
  __read_only image2d_t weights_zero_point_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  int dst_s = get_global_id(0);
  int dst_end_slice = shared_int4_0.x;
  int dst_s_wg_offset = get_group_id(0) * get_local_size(0);
  if (dst_s_wg_offset >= dst_end_slice) return;
  half4 r_sp0_s0 = (half4)(0.0f);
  int2 tid;
  tid.x = get_local_id(0);
  tid.y = get_local_id(1);
  if (dst_s < shared_int4_0.x) {
  for (int src_group = tid.y; src_group < shared_int4_0.z; src_group += 16) {
    half4 w_scale_s0 = read_imageh(weights_scale_image2d, smp_zero, (int2)((dst_s), ((0) * shared_int4_1.x + (src_group))));
    half4 w_zp_s0 = read_imageh(weights_zero_point_image2d, smp_zero, (int2)((dst_s), ((0) * shared_int4_1.y + (src_group))));
    half4 w_bias_s0 = -w_scale_s0 * ((half4)(8) + w_zp_s0);
  for (int src_sub_id = 0; src_sub_id < shared_int4_0.y; src_sub_id += 1) {
    int src_s = src_group * shared_int4_0.y + src_sub_id;
    half4 v0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_0.w + (src_s))));
    half4 w0, w1, w2, w3;
    ushort4 w = convert_ushort4(read_imageui(weights_image2d, smp_zero, (int2)((dst_s), (src_s))));
    
  w0.x = convert_half((w.x) & 15u);
  w0.y = convert_half((w.x >>  4u) & 15u);
  w0.z = convert_half((w.x >>  8u) & 15u);
  w0.w = convert_half((w.x >> 12u) & 15u);
  w1.x = convert_half((w.y) & 15u);
  w1.y = convert_half((w.y >> 4u) & 15u);
  w1.z = convert_half((w.y >> 8u) & 15u);
  w1.w = convert_half((w.y >> 12u) & 15u);
  w2.x = convert_half((w.z) & 15u);
  w2.y = convert_half((w.z >>  4u) & 15u);
  w2.z = convert_half((w.z >>  8u) & 15u);
  w2.w = convert_half((w.z >> 12u) & 15u);
  w3.x = convert_half((w.w) & 15u);
  w3.y = convert_half((w.w >> 4u) & 15u);
  w3.z = convert_half((w.w >> 8u) & 15u);
  w3.w = convert_half((w.w >> 12u) & 15u);
;
    w0 = w0 * w_scale_s0 + w_bias_s0;
    w1 = w1 * w_scale_s0 + w_bias_s0;
    w2 = w2 * w_scale_s0 + w_bias_s0;
    w3 = w3 * w_scale_s0 + w_bias_s0;
    r_sp0_s0 += v0.x * w0;
    r_sp0_s0 += v0.y * w1;
    r_sp0_s0 += v0.z * w2;
    r_sp0_s0 += v0.w * w3;
  }} 
  } 
  __local half4 temp[64];
  temp[tid.x * 16 + tid.y] = r_sp0_s0;
  for (int ystride = 16 / 2; ystride > 0; ystride /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid.y < ystride) {
      r_sp0_s0 += temp[tid.x * 16 + tid.y + ystride];
      temp[tid.x * 16 + tid.y] = r_sp0_s0;
    }
  }
  if (dst_s >= shared_int4_0.x) return;
  if (tid.y != 0) return;
  {
  half4 res_value = convert_half4(r_sp0_s0);
  write_imageh(dst_tensor_image2d, (int2)((0), ((0) * shared_int4_0.x + (dst_s + 0))), res_value);
  }
}

