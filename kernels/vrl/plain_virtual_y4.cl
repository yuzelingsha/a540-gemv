#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define ADRENO_ACCUM_VLANE(ACC, VY) do { \
  for (int src_group = (VY); src_group < shared_int4_0.z; src_group += 16) { \
    half4 w_scale_s0 = read_imageh(weights_scale_image2d, smp_zero, (int2)(dst_s, src_group)); \
    half4 adrenollm_block_acc = (half4)(0.0f); \
    for (int src_sub_id = 0; src_sub_id < shared_int4_0.y; src_sub_id += 1) { \
      int src_s = src_group * shared_int4_0.y + src_sub_id; \
      half4 v0 = read_imageh(src_tensor_image2d, smp_zero, (int2)(0, src_s)); \
      half4 w0, w1, w2, w3; \
      ushort4 w = convert_ushort4(read_imageui(weights_image2d, smp_zero, (int2)(dst_s, src_s))); \
      w0.x = convert_half((w.x) & 15u); \
      w0.y = convert_half((w.x >> 4u) & 15u); \
      w0.z = convert_half((w.x >> 8u) & 15u); \
      w0.w = convert_half((w.x >> 12u) & 15u); \
      w1.x = convert_half((w.y) & 15u); \
      w1.y = convert_half((w.y >> 4u) & 15u); \
      w1.z = convert_half((w.y >> 8u) & 15u); \
      w1.w = convert_half((w.y >> 12u) & 15u); \
      w2.x = convert_half((w.z) & 15u); \
      w2.y = convert_half((w.z >> 4u) & 15u); \
      w2.z = convert_half((w.z >> 8u) & 15u); \
      w2.w = convert_half((w.z >> 12u) & 15u); \
      w3.x = convert_half((w.w) & 15u); \
      w3.y = convert_half((w.w >> 4u) & 15u); \
      w3.z = convert_half((w.w >> 8u) & 15u); \
      w3.w = convert_half((w.w >> 12u) & 15u); \
      adrenollm_block_acc += v0.x * w0; \
      adrenollm_block_acc += v0.y * w1; \
      adrenollm_block_acc += v0.z * w2; \
      adrenollm_block_acc += v0.w * w3; \
      adrenollm_block_acc -= (half4)(8) * (v0.x + v0.y + v0.z + v0.w); \
    } \
    (ACC) += adrenollm_block_acc * w_scale_s0; \
  } \
} while (0)

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
  int2 tid;
  tid.x = get_local_id(0);
  tid.y = get_local_id(1);

  half4 r0 = (half4)(0.0f);
  half4 r4 = (half4)(0.0f);
  half4 r8 = (half4)(0.0f);
  half4 r12 = (half4)(0.0f);
  if (dst_s < shared_int4_0.x) {
    ADRENO_ACCUM_VLANE(r0, tid.y);
    ADRENO_ACCUM_VLANE(r4, tid.y + 4);
    ADRENO_ACCUM_VLANE(r8, tid.y + 8);
    ADRENO_ACCUM_VLANE(r12, tid.y + 12);
  }

  __local half4 temp[256];
  int base = tid.x * 16 + tid.y;
  temp[base] = r0;
  temp[base + 4] = r4;
  temp[base + 8] = r8;
  temp[base + 12] = r12;

  // Reproduce the original 16-lane reduction tree exactly: 8 -> 4 -> 2 -> 1.
  barrier(CLK_LOCAL_MEM_FENCE);
  temp[base] = temp[base] + temp[base + 8];
  temp[base + 4] = temp[base + 4] + temp[base + 12];

  barrier(CLK_LOCAL_MEM_FENCE);
  temp[base] = temp[base] + temp[base + 4];

  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid.y < 2) temp[base] = temp[base] + temp[base + 2];

  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid.y == 0) temp[base] = temp[base] + temp[base + 1];

  if (dst_s >= shared_int4_0.x) return;
  if (tid.y != 0) return;
  half4 res_value = convert_half4(temp[tid.x * 16]);
  write_imageh(dst_tensor_image2d, (int2)(0, dst_s), res_value);
}
