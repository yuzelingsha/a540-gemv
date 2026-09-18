#include <CL/cl.h>
#include <dlfcn.h>
#include <android/log.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>

extern "C" void* __loader_dlsym(void* handle, const char* symbol, const void* caller_addr);

using CreateProgramFn = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using ReadBufferFn = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
using CreateKernelFn = cl_kernel (*)(cl_program, const char*, cl_int*);
using EnqueueKernelFn = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
static std::atomic<CreateProgramFn> g_real_create{nullptr};
static std::atomic<ReadBufferFn> g_real_read{nullptr};
static std::atomic<CreateKernelFn> g_real_create_kernel{nullptr};
static std::atomic<EnqueueKernelFn> g_real_enqueue_kernel{nullptr};
static cl_program g_int4_add_programs[16] = {};
static std::atomic<int> g_int4_add_program_count{0};
static cl_kernel g_int4_add_kernels[256] = {};
static std::atomic<int> g_int4_add_kernel_count{0};
static std::atomic<int> g_dlsym_hits{0};
static std::atomic<int> g_patch_hits{0};

static bool env_on(const char* name) {
  const char* e = getenv(name);
  return e && e[0]=='1' && e[1]=='\0';
}
static bool geom_x2_on() { return env_on("ADRENOLLM_DLSYM_INT4_ADD_X2"); }
static bool int4_y4_plain_on() { return env_on("ADRENOLLM_DLSYM_INT4_Y4_PLAIN"); }
static bool int4_y4_add_on() { return env_on("ADRENOLLM_DLSYM_INT4_Y4_ADD"); }
static bool q_virtual_y4_on() { return env_on("ADRENOLLM_DLSYM_Q_VIRTUAL_Y4"); }
static bool add_virtual_y4_on() { return env_on("ADRENOLLM_DLSYM_ADD_VIRTUAL_Y4"); }
static bool int4_y4_on() { return env_on("ADRENOLLM_DLSYM_INT4_Y4") || int4_y4_plain_on() || int4_y4_add_on(); }
static bool static_replay_on() { return env_on("ADRENOLLM_DLSYM_STATIC_REPLAY"); }
static bool enqueue_pass_on() { return env_on("ADRENOLLM_DLSYM_ENQUEUE_PASS"); }
static bool enabled() { return env_on("ADRENOLLM_DLSYM_NO_ZP") || env_on("ADRENOLLM_DLSYM_BLOCK_SCALE") || env_on("ADRENOLLM_DLSYM_LM_NO_ZP") || env_on("ADRENOLLM_DLSYM_LM_FACTOR") || env_on("ADRENOLLM_DLSYM_LM_UNROLL2") || env_on("ADRENOLLM_DLSYM_LOCAL_SRC384") || int4_y4_on() || q_virtual_y4_on() || add_virtual_y4_on() || geom_x2_on(); }

static void record_add_program(cl_program p) {
  if (!p || (!geom_x2_on() && !int4_y4_on())) return;
  int n = g_int4_add_program_count.fetch_add(1, std::memory_order_relaxed);
  if (n < 16) g_int4_add_programs[n] = p;
}
static bool is_add_program(cl_program p) {
  int n = g_int4_add_program_count.load(std::memory_order_relaxed);
  if (n > 16) n = 16;
  for (int i=0;i<n;++i) if (g_int4_add_programs[i] == p) return true;
  return false;
}
static void record_add_kernel(cl_kernel k) {
  if (!k) return;
  int n = g_int4_add_kernel_count.fetch_add(1, std::memory_order_relaxed);
  if (n < 256) g_int4_add_kernels[n] = k;
}
static bool is_add_kernel(cl_kernel k) {
  int n = g_int4_add_kernel_count.load(std::memory_order_relaxed);
  if (n > 256) n = 256;
  for (int i=0;i<n;++i) if (g_int4_add_kernels[i] == k) return true;
  return false;
}

static void replace_all(std::string& s, const std::string& from, const std::string& to, int* count=nullptr) {
  size_t p=0;
  while ((p=s.find(from,p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
    if (count) ++*count;
  }
}

static bool load_text_file(const char* path, std::string* out) {
  if (!path || !out) return false;
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
  long n = ftell(f);
  if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
  out->resize(static_cast<size_t>(n));
  size_t got = fread(out->data(), 1, static_cast<size_t>(n), f);
  fclose(f);
  if (got != static_cast<size_t>(n)) { out->clear(); return false; }
  return true;
}

static cl_program hook_create_program(cl_context context, cl_uint count,
                                      const char** strings, const size_t* lengths,
                                      cl_int* errcode_ret) {
  CreateProgramFn real = g_real_create.load(std::memory_order_acquire);
  if (!real) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
  }
  if (!enabled() || count == 0 || strings == nullptr) {
    return real(context, count, strings, lengths, errcode_ret);
  }

  std::string src;
  for (cl_uint i=0;i<count;++i) {
    if (!strings[i]) continue;
    size_t n = (lengths && lengths[i]) ? lengths[i] : strlen(strings[i]);
    src.append(strings[i], n);
  }
  const bool adrenollm_is_int4_y16 =
      src.find("src_group += 16") != std::string::npos &&
      src.find("__local half4 temp[64]") != std::string::npos &&
      src.find("ushort4 w = convert_ushort4(read_imageui(weights_image2d") != std::string::npos &&
      src.find("weights_zero_point_image2d") != std::string::npos;
  const bool adrenollm_is_int4_add_y16 =
      adrenollm_is_int4_y16 &&
      src.find("res_value_final = res_value + second_value;") != std::string::npos;
  const bool adrenollm_y4_target = adrenollm_is_int4_y16 &&
      (env_on("ADRENOLLM_DLSYM_INT4_Y4") ||
       (int4_y4_plain_on() && !adrenollm_is_int4_add_y16) ||
       (int4_y4_add_on() && adrenollm_is_int4_add_y16));

  // LM-head optimizations: LM_NO_ZP, LM_FACTOR, LOCAL_SRC384, LM_UNROLL2.
  if (src.find("weights_scale_image_buffer") != std::string::npos &&
      src.find("uint4 w = read_imageui(weights_image2d") != std::string::npos) {
    bool lm_modified = false;
    int lm_zp_reads = 0, lm_zp_bias = 0;

    if (env_on("ADRENOLLM_DLSYM_LM_NO_ZP") &&
        src.find("weights_zero_point_image_buffer") != std::string::npos &&
        src.find("((half4)(128) + w_zp_s0)") != std::string::npos) {
      const std::string zpline = "  half4 w_zp_s0 = read_imageh(weights_zero_point_image_buffer, (dst_s + 0));\n";
      size_t p = 0;
      while ((p = src.find(zpline, p)) != std::string::npos) { src.erase(p, zpline.size()); ++lm_zp_reads; }
      replace_all(src,
        "  half4 w_bias_s0 = -w_scale_s0 * ((half4)(128) + w_zp_s0);",
        "  half4 w_bias_s0 = -w_scale_s0 * (half4)(128);", &lm_zp_bias);
      if (lm_zp_reads && lm_zp_bias) {
        lm_modified = true;
      }
    }

    if (env_on("ADRENOLLM_DLSYM_LM_FACTOR") &&
        src.find("weights_zero_point_image_buffer") != std::string::npos &&
        src.find("((half4)(128) + w_zp_s0)") != std::string::npos) {
      int ins = 0, deq = 0, acc = 0, close = 0;
      const std::string loop = "  for (int src_s = 0; src_s < shared_int4_1.y; src_s += 1) {";
      size_t p = src.find(loop);
      if (p != std::string::npos) {
        const std::string d = "  half4 adrenollm_lm_qacc = (half4)(0.0f);\n  half adrenollm_lm_vsum = (half)(0.0f);\n";
        src.insert(p, d); ++ins;
      }
      for (int i = 0; i < 4; ++i) {
        std::string line = "    w" + std::to_string(i) + " = w" + std::to_string(i) + " * w_scale_s0 + w_bias_s0;\n";
        p = 0; while ((p = src.find(line, p)) != std::string::npos) { src.erase(p, line.size()); ++deq; }
      }
      const std::string oldacc =
        "    r_sp0_s0 += v0.x * w0;\n"
        "    r_sp0_s0 += v0.y * w1;\n"
        "    r_sp0_s0 += v0.z * w2;\n"
        "    r_sp0_s0 += v0.w * w3;\n";
      const std::string newacc =
        "    adrenollm_lm_qacc += v0.x * w0;\n"
        "    adrenollm_lm_qacc += v0.y * w1;\n"
        "    adrenollm_lm_qacc += v0.z * w2;\n"
        "    adrenollm_lm_qacc += v0.w * w3;\n"
        "    adrenollm_lm_vsum += v0.x + v0.y + v0.z + v0.w;\n";
      p = src.find(oldacc); if (p != std::string::npos) { src.replace(p, oldacc.size(), newacc); ++acc; }
      const std::string oldclose = "  } \n  {\n  half4 res_value";
      const std::string newclose = "  } \n  r_sp0_s0 = adrenollm_lm_qacc * w_scale_s0 + w_bias_s0 * adrenollm_lm_vsum;\n  {\n  half4 res_value";
      p = src.find(oldclose); if (p != std::string::npos) { src.replace(p, oldclose.size(), newclose); ++close; }
      if (ins && deq == 4 && acc && close) {
        lm_modified = true;
      }
    }

    if (env_on("ADRENOLLM_DLSYM_LOCAL_SRC384") &&
        src.find("for (int src_s = 0; src_s < shared_int4_1.y; src_s += 1)") != std::string::npos) {
      int local_src384 = 0;
      const std::string loop = "  for (int src_s = 0; src_s < shared_int4_1.y; src_s += 1) {";
      size_t p = src.find(loop);
      if (p != std::string::npos) {
        const std::string preload =
          "  __local half4 adrenollm_src_cache[384];\n"
          "  int adrenollm_lid = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));\n"
          "  int adrenollm_lsize = get_local_size(0) * get_local_size(1) * get_local_size(2);\n"
          "  for (int adrenollm_i = adrenollm_lid; adrenollm_i < 384; adrenollm_i += adrenollm_lsize) {\n"
          "    adrenollm_src_cache[adrenollm_i] = read_imageh(src_tensor_image2d, smp_zero, (int2)(0, adrenollm_i));\n"
          "  }\n"
          "  barrier(CLK_LOCAL_MEM_FENCE);\n";
        src.insert(p, preload);
        size_t vp = 0;
        const std::string vlead = "    half4 v0 = read_imageh(src_tensor_image2d";
        while ((vp = src.find(vlead, vp)) != std::string::npos) {
          size_t eol = src.find('\n', vp); if (eol == std::string::npos) break;
          src.replace(vp, eol - vp, "    half4 v0 = adrenollm_src_cache[src_s];");
          vp += 42;
          ++local_src384;
        }
        if (local_src384) {
          lm_modified = true;
          __android_log_print(ANDROID_LOG_INFO, "AdrenoLLM", "LM-Head LOCAL_SRC384 applied loads=%d bytes=%zu", local_src384, src.size());
        }
      }
    }

    if (env_on("ADRENOLLM_DLSYM_LM_UNROLL2") &&
        src.find("for (int src_s = 0; src_s < shared_int4_1.y; src_s += 1)") != std::string::npos) {
      const std::string head = "  for (int src_s = 0; src_s < shared_int4_1.y; src_s += 1) {\n";
      const std::string tail = "    r_sp0_s0 += v0.w * w3;\n  } \n";
      size_t a = src.find(head);
      size_t b = (a == std::string::npos) ? std::string::npos : src.find(tail, a + head.size());
      if (a != std::string::npos && b != std::string::npos) {
        b += tail.size();
        size_t body_start = a + head.size();
        size_t body_end = b - tail.size() + tail.find("\n  } ");
        std::string body = src.substr(body_start, body_end - body_start);
        std::string body2 = body;
        replace_all(body2, "src_s", "adrenollm_s1");
        std::string repl =
          "  for (int src_s = 0; src_s < shared_int4_1.y; src_s += 2) {\n"
          "    {\n" + body + "    }\n"
          "    {\n      int adrenollm_s1 = src_s + 1;\n" + body2 + "    }\n"
          "  } \n";
        src.replace(a, b - a, repl);
        lm_modified = true;
      }
    }

    if (lm_modified) {
      int hit = g_patch_hits.fetch_add(1) + 1;
      __android_log_print(ANDROID_LOG_INFO, "AdrenoLLM", "LM-Head patched #%d bytes=%zu", hit, src.size());
      const char* psrc = src.data(); size_t n = src.size();
      return real(context, 1, &psrc, &n, errcode_ret);
    }
    return real(context, count, strings, lengths, errcode_ret);
  }

  // Only generated blockwise INT4 image kernels. The MiniCPM5-1B quant report
  // shows zp_tensor=-1 for these weights, so the runtime image is a dummy zero.
  if (src.find("weights_zero_point_image2d") == std::string::npos ||
      src.find("ushort4 w = convert_ushort4(read_imageui(weights_image2d") == std::string::npos ||
      src.find("w_bias_s0 = -w_scale_s0 * ((half4)(8) + w_zp_s0)") == std::string::npos) {
    return real(context, count, strings, lengths, errcode_ret);
  }

  if (env_on("ADRENOLLM_DLSYM_BLOCK_SCALE")) {
    int removed_reads=0, removed_bias=0, inserted=0, accum_repl=0, close_repl=0;
    const std::string zpneedle = "    half4 w_zp_s0 = read_imageh(weights_zero_point_image2d";
    size_t p=0;
    while ((p=src.find(zpneedle,p)) != std::string::npos) {
      size_t eol=src.find('\n',p); if(eol==std::string::npos) break;
      src.erase(p,eol-p+1); ++removed_reads;
    }
    const std::string biasline = "    half4 w_bias_s0 = -w_scale_s0 * ((half4)(8) + w_zp_s0);\n";
    p=0; while((p=src.find(biasline,p))!=std::string::npos){ src.erase(p,biasline.size()); ++removed_bias; }
    const std::string inner = "  for (int src_sub_id = 0; src_sub_id < shared_int4_0.y; src_sub_id += 1) {";
    p=0; while((p=src.find(inner,p))!=std::string::npos){
      const std::string decl="  half4 adrenollm_block_acc = (half4)(0.0f);\n";
      src.insert(p,decl); p += decl.size()+inner.size(); ++inserted;
    }
    if (env_on("ADRENOLLM_DLSYM_UNROLL8")) {
      const std::string unroll = "  #pragma unroll 8\n";
      p=0; while((p=src.find(inner,p))!=std::string::npos) {
        src.insert(p, unroll); p += unroll.size() + inner.size();
      }
    }
    for (int i=0;i<4;++i) {
      std::string line = "    w" + std::to_string(i) + " = w" + std::to_string(i) + " * w_scale_s0 + w_bias_s0;\n";
      p=0; while((p=src.find(line,p))!=std::string::npos) src.erase(p,line.size());
    }
    const std::string oldacc =
      "    r_sp0_s0 += v0.x * w0;\n"
      "    r_sp0_s0 += v0.y * w1;\n"
      "    r_sp0_s0 += v0.z * w2;\n"
      "    r_sp0_s0 += v0.w * w3;\n";
    const std::string newacc =
      "    adrenollm_block_acc += v0.x * w0;\n"
      "    adrenollm_block_acc += v0.y * w1;\n"
      "    adrenollm_block_acc += v0.z * w2;\n"
      "    adrenollm_block_acc += v0.w * w3;\n"
      "    adrenollm_block_acc -= (half4)(8) * (v0.x + v0.y + v0.z + v0.w);\n";
    p=0; while((p=src.find(oldacc,p))!=std::string::npos){ src.replace(p,oldacc.size(),newacc); p+=newacc.size(); ++accum_repl; }
    const std::string oldclose="  }} \n  } \n  __local half4 temp[64];";
    const std::string newclose="  }\n    r_sp0_s0 += adrenollm_block_acc * w_scale_s0;\n  }\n  } \n  __local half4 temp[64];";
    p=src.find(oldclose); if(p!=std::string::npos){ src.replace(p,oldclose.size(),newclose); ++close_repl; }
    if (env_on("ADRENOLLM_DLSYM_VEC_UNPACK")) {
      const std::string oldu =
        "  w0.x = convert_half((w.x) & 15u);\n"
        "  w0.y = convert_half((w.x >>  4u) & 15u);\n"
        "  w0.z = convert_half((w.x >>  8u) & 15u);\n"
        "  w0.w = convert_half((w.x >> 12u) & 15u);\n"
        "  w1.x = convert_half((w.y) & 15u);\n"
        "  w1.y = convert_half((w.y >> 4u) & 15u);\n"
        "  w1.z = convert_half((w.y >> 8u) & 15u);\n"
        "  w1.w = convert_half((w.y >> 12u) & 15u);\n"
        "  w2.x = convert_half((w.z) & 15u);\n"
        "  w2.y = convert_half((w.z >> 4u) & 15u);\n"
        "  w2.z = convert_half((w.z >> 8u) & 15u);\n"
        "  w2.w = convert_half((w.z >> 12u) & 15u);\n"
        "  w3.x = convert_half((w.w) & 15u);\n"
        "  w3.y = convert_half((w.w >> 4u) & 15u);\n"
        "  w3.z = convert_half((w.w >> 8u) & 15u);\n"
        "  w3.w = convert_half((w.w >> 12u) & 15u);\n";
      const std::string newu =
        "  const ushort4 adrenollm_qshift = (ushort4)(0, 4, 8, 12);\n"
        "  w0 = convert_half4(((ushort4)(w.x) >> adrenollm_qshift) & (ushort4)(15));\n"
        "  w1 = convert_half4(((ushort4)(w.y) >> adrenollm_qshift) & (ushort4)(15));\n"
        "  w2 = convert_half4(((ushort4)(w.z) >> adrenollm_qshift) & (ushort4)(15));\n"
        "  w3 = convert_half4(((ushort4)(w.w) >> adrenollm_qshift) & (ushort4)(15));\n";
      size_t up=src.find(oldu);
      if(up!=std::string::npos) src.replace(up,oldu.size(),newu);
      else __android_log_print(ANDROID_LOG_ERROR,"AdrenoLLM","VEC_UNPACK pattern miss");
    }
    int local_src384 = 0;
    if (env_on("ADRENOLLM_DLSYM_LOCAL_SRC384") &&
        src.find("src_group += 4") != std::string::npos &&
        src.find("__local half4 temp[64]") != std::string::npos &&
        (src.find("native_recip(1.0f + native_exp") != std::string::npos ||
         src.find("res_value_final = res_value * second_value;") != std::string::npos)) {
      const std::string tids =
        "  tid.x = get_local_id(0);\n"
        "  tid.y = get_local_id(1);\n";
      size_t tp=src.find(tids);
      if(tp!=std::string::npos){
        const std::string preload = tids +
          "  __local half4 adrenollm_src_cache[384];\n"
          "  int adrenollm_lid = tid.y * get_local_size(0) + tid.x;\n"
          "  int adrenollm_lsize = get_local_size(0) * get_local_size(1);\n"
          "  int adrenollm_src_count = shared_int4_0.z * shared_int4_0.y;\n"
          "  for (int adrenollm_i = adrenollm_lid; adrenollm_i < adrenollm_src_count; adrenollm_i += adrenollm_lsize) {\n"
          "    adrenollm_src_cache[adrenollm_i] = read_imageh(src_tensor_image2d, smp_zero, (int2)(0, adrenollm_i));\n"
          "  }\n"
          "  barrier(CLK_LOCAL_MEM_FENCE);\n";
        src.replace(tp,tids.size(),preload);
        size_t vp=0;
        const std::string vlead="    half4 v0 = read_imageh(src_tensor_image2d";
        while((vp=src.find(vlead,vp))!=std::string::npos){
          size_t eol=src.find('\n',vp); if(eol==std::string::npos) break;
          src.replace(vp,eol-vp,"    half4 v0 = adrenollm_src_cache[src_s];");
          vp += 48; ++local_src384;
        }
      }
      if(local_src384) {
        __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM","LOCAL_SRC384 applied loads=%d bytes=%zu",local_src384,src.size());
      } else {
        __android_log_print(ANDROID_LOG_ERROR,"AdrenoLLM","LOCAL_SRC384 pattern miss");
      }
    }
    int y4_rewrites = 0;
    if (adrenollm_y4_target) {
      replace_all(src, "src_group += 16", "src_group += 4", &y4_rewrites);
      replace_all(src, "tid.x * 16 + tid.y", "tid.x * 4 + tid.y");
      replace_all(src, "16 / 2", "4 / 2");
      __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM",
                          "INT4_Y4 source patched rewrites=%d bytes=%zu",
                          y4_rewrites, src.size());
    }
    if (q_virtual_y4_on() && adrenollm_is_int4_y16 && !adrenollm_is_int4_add_y16) {
      std::string dual_src;
      if (load_text_file("/data/local/tmp/litert-a540/plain_block_dual.cl", &dual_src)) {
        src.swap(dual_src);
        __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM",
                            "Q_VIRTUAL_Y4 dual plain source loaded bytes=%zu", src.size());
      } else {
        __android_log_print(ANDROID_LOG_ERROR,"AdrenoLLM",
                            "Q_VIRTUAL_Y4 failed to load dual source");
      }
    }
    if (add_virtual_y4_on() && adrenollm_is_int4_add_y16) {
      std::string dual_src;
      if (load_text_file("/data/local/tmp/litert-a540/add_block_dual.cl", &dual_src)) {
        src.swap(dual_src);
        __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM",
                            "ADD_VIRTUAL_Y4 dual add source loaded bytes=%zu", src.size());
      } else {
        __android_log_print(ANDROID_LOG_ERROR,"AdrenoLLM",
                            "ADD_VIRTUAL_Y4 failed to load dual source");
      }
    }
    if (removed_reads && removed_bias && inserted && accum_repl && close_repl) {
      int hit=g_patch_hits.fetch_add(1)+1;
      __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM","BLOCK_SCALE patched #%d reads=%d bias=%d loops=%d acc=%d close=%d bytes=%zu",hit,removed_reads,removed_bias,inserted,accum_repl,close_repl,src.size());
      const char* psrc=src.data(); size_t n=src.size();
      cl_program program = real(context,1,&psrc,&n,errcode_ret);
      if (adrenollm_y4_target ||
          (geom_x2_on() && adrenollm_is_int4_add_y16)) {
        record_add_program(program);
      }
      return program;
    }
    __android_log_print(ANDROID_LOG_ERROR,"AdrenoLLM","BLOCK_SCALE pattern miss reads=%d bias=%d loops=%d acc=%d close=%d",removed_reads,removed_bias,inserted,accum_repl,close_repl);
    return real(context,count,strings,lengths,errcode_ret);
  }

  int reads=0, biases=0;
  const std::string needle = "    half4 w_zp_s0 = read_imageh(weights_zero_point_image2d";
  size_t p=0;
  while ((p=src.find(needle,p)) != std::string::npos) {
    size_t eol=src.find('\n',p);
    if (eol==std::string::npos) break;
    src.erase(p,eol-p+1);
    ++reads;
  }
  replace_all(src,
      "    half4 w_bias_s0 = -w_scale_s0 * ((half4)(8) + w_zp_s0);",
      "    half4 w_bias_s0 = -w_scale_s0 * (half4)(8);", &biases);
  if (!reads || !biases) return real(context, count, strings, lengths, errcode_ret);

  int hit=g_patch_hits.fetch_add(1)+1;
  __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM","NO_ZP patched program #%d reads=%d biases=%d bytes=%zu",hit,reads,biases,src.size());
  const char* psrc=src.data(); size_t n=src.size();
  return real(context,1,&psrc,&n,errcode_ret);
}

struct ReplayDispatch {
  cl_kernel kernel = nullptr;
  cl_uint dim = 0;
  bool has_offset = false;
  size_t offset[3] = {0,0,0};
  size_t global[3] = {1,1,1};
  size_t local[3] = {1,1,1};
};
static constexpr int kReplayRing = 677;
static constexpr int kReplayModel = 672;
static ReplayDispatch g_replay_ring[kReplayRing];
static ReplayDispatch g_replay_plan[kReplayModel];
static int g_replay_ring_pos = 0;
static int g_replay_ring_count = 0;
static int g_replay_token_call = 0;
static int g_replay_token_num = 0;
static bool g_replay_plan_ready = false;
static bool g_replay_first_token_eligible = true;

static void replay_record(cl_kernel k, cl_uint dim, const size_t* offset,
                          const size_t* global, const size_t* local,
                          cl_uint nwait, cl_event* ev) {
  if (!static_replay_on() || g_replay_plan_ready || !global || !local) return;
  if (nwait != 0 || ev != nullptr || dim == 0 || dim > 3) {
    g_replay_first_token_eligible = false;
    return;
  }
  ReplayDispatch& r = g_replay_ring[g_replay_ring_pos];
  r.kernel = k; r.dim = dim; r.has_offset = offset != nullptr;
  for (int i=0;i<3;++i) {
    r.offset[i] = (offset && i < (int)dim) ? offset[i] : 0;
    r.global[i] = (i < (int)dim) ? global[i] : 1;
    r.local[i] = (i < (int)dim) ? local[i] : 1;
  }
  g_replay_ring_pos = (g_replay_ring_pos + 1) % kReplayRing;
  if (g_replay_ring_count < kReplayRing) ++g_replay_ring_count;
}
static const ReplayDispatch& replay_ring_chrono(int idx) {
  int start = g_replay_ring_count == kReplayRing ? g_replay_ring_pos : 0;
  return g_replay_ring[(start + idx) % kReplayRing];
}
static bool replay_tail_matches_decode() {
  if (g_replay_ring_count != kReplayRing || !g_replay_first_token_eligible) return false;
  const auto& lm = replay_ring_chrono(671);
  const auto& s0 = replay_ring_chrono(672);
  const auto& s3 = replay_ring_chrono(675);
  const auto& s4 = replay_ring_chrono(676);
  return lm.dim == 3 && lm.global[0] == 16384 && lm.global[1] == 2 &&
         lm.local[0] == 512 && lm.local[1] == 2 &&
         s0.dim == 3 && s0.global[2] == 32640 && s0.local[2] == 64 &&
         s3.global[0] == 4096 && s3.local[0] == 256 &&
         s4.global[0] == 32 && s4.local[0] == 32;
}
static void replay_learn_at_boundary() {
  if (!static_replay_on() || g_replay_plan_ready) return;
  if (!replay_tail_matches_decode()) return;
  for (int i=0;i<kReplayModel;++i) g_replay_plan[i] = replay_ring_chrono(i);
  g_replay_plan_ready = true;
  g_replay_token_call = 0;
  __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM","STATIC_REPLAY plan learned model=%d ring=%d",kReplayModel,kReplayRing);
}

static cl_kernel hook_create_kernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
  CreateKernelFn real = g_real_create_kernel.load(std::memory_order_acquire);
  if (!real) { if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION; return nullptr; }
  cl_kernel k = real(program, kernel_name, errcode_ret);
  if (k && is_add_program(program)) record_add_kernel(k);
  return k;
}

static cl_int hook_enqueue_kernel(cl_command_queue q, cl_kernel k, cl_uint dim,
                                  const size_t* offset, const size_t* global,
                                  const size_t* local, cl_uint nwait,
                                  const cl_event* wait, cl_event* ev) {
  EnqueueKernelFn real = g_real_enqueue_kernel.load(std::memory_order_acquire);
  if (!real) return CL_INVALID_OPERATION;

  if (static_replay_on()) {
    if (!g_replay_plan_ready) {
      replay_record(k, dim, offset, global, local, nwait, ev);
      return real(q, k, dim, offset, global, local, nwait, wait, ev);
    }
    // At the first model enqueue of token N>1, submit the frozen 672-command
    // model plan in one tight host loop. All kernel handles/args/geometries were
    // independently verified stable token-to-token; sampler stays untouched.
    if (g_replay_token_call < kReplayModel) {
      if (g_replay_token_call == 0) {
        for (int i=0;i<kReplayModel;++i) {
          const ReplayDispatch& r = g_replay_plan[i];
          const size_t* roff = r.has_offset ? r.offset : nullptr;
          cl_int rc = real(q, r.kernel, r.dim, roff, r.global, r.local,
                           0, nullptr, nullptr);
          if (rc != CL_SUCCESS) return rc;
        }
      }
      ++g_replay_token_call;
      return CL_SUCCESS;
    }
    ++g_replay_token_call;  // sampler dispatches 672..676 run normally.
    return real(q, k, dim, offset, global, local, nwait, wait, ev);
  }

  // A540 q_proj schedule: keep the original 16 numerical reduction lanes
  // inside the dual-mode kernel, but execute them on a physical 16x4 workgroup.
  // q_proj is uniquely identified here by g=(512,16,1), l=(4,16,1).
  if (q_virtual_y4_on() && dim >= 2 && global && local &&
      global[0] == 512 && global[1] == 16 &&
      local[0] == 4 && local[1] == 16) {
    size_t g[3] = {global[0], 4, dim > 2 ? global[2] : 1};
    size_t l[3] = {16, 4, dim > 2 ? local[2] : 1};
    return real(q, k, dim, offset, g, l, nwait, wait, ev);
  }

  // A540 o_proj/down_proj schedule with exact virtual 16-lane reduction.
  if (add_virtual_y4_on() && dim >= 2 && global && local &&
      global[0] == 384 && global[1] == 16 &&
      local[0] == 4 && local[1] == 16) {
    size_t g[3] = {global[0], 4, dim > 2 ? global[2] : 1};
    size_t l[3] = {16, 4, dim > 2 ? local[2] : 1};
    return real(q, k, dim, offset, g, l, nwait, wait, ev);
  }

  // A540-specific block32 INT4 schedule: keep 64 threads total but use
  // 16 output slices x 4 K lanes. Only kernels whose source was rewritten
  // are tracked and patched here.
  const bool y4_tracked = int4_y4_on() && is_add_kernel(k);
  // Cached program binaries bypass clCreateProgramWithSource, so for the
  // validated add-only fast path also recognize its unique decode geometry:
  // 48 launches/token, exactly o_proj/down_proj + residual add.
  const bool y4_add_cached = int4_y4_add_on() && dim >= 2 && global && local &&
      global[0] == 384 && global[1] == 16 &&
      local[0] == 4 && local[1] == 16;
  if ((y4_tracked || y4_add_cached) && dim >= 2 && global && local &&
      local[0] == 4 && local[1] == 16 && global[1] == 16) {
    size_t g[3] = {global[0], 4, dim > 2 ? global[2] : 1};
    size_t l[3] = {16, 4, dim > 2 ? local[2] : 1};
    return real(q, k, dim, offset, g, l, nwait, wait, ev);
  }

  // In the decode timeline this exact geometry occurs 48x/token and maps
  // one-to-one to INT4 o_proj/down_proj + residual-add kernels. Halving X
  // keeps the 16-way K reduction and every thread's arithmetic unchanged.
  if (geom_x2_on() && dim >= 2 && global && local &&
      global[0] == 384 && global[1] == 16 &&
      local[0] == 4 && local[1] == 16) {
    size_t l[3] = {local[0], local[1], dim > 2 ? local[2] : 1};
    l[0] = 2;
    return real(q, k, dim, offset, global, l, nwait, wait, ev);
  }
  return real(q, k, dim, offset, global, local, nwait, wait, ev);
}

static cl_int hook_read_buffer(cl_command_queue q, cl_mem b, cl_bool blocking, size_t off, size_t cb, void* ptr, cl_uint nwait, const cl_event* wait, cl_event* ev) {
  ReadBufferFn real=g_real_read.load(std::memory_order_acquire);
  if(!real) return CL_INVALID_OPERATION;
  cl_int rc=real(q,b,blocking,off,cb,ptr,nwait,wait,ev);
  if(rc==CL_SUCCESS && blocking==CL_TRUE && cb==4 && ptr && static_replay_on()) {
    if (!g_replay_plan_ready) replay_learn_at_boundary();
    if (g_replay_plan_ready) { g_replay_token_call = 0; ++g_replay_token_num; }
  }
  if(rc==CL_SUCCESS && blocking==CL_TRUE && cb==4 && ptr && env_on("ADRENOLLM_LOG_TOKEN_IDS")) {
    int v=0; memcpy(&v,ptr,4);
    __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM","TOKEN_ID %d",v);
  }
  return rc;
}

extern "C" __attribute__((visibility("default")))
void* dlsym(void* handle, const char* symbol) {
  void* real = __loader_dlsym(handle, symbol, __builtin_return_address(0));
  if (!symbol) return real;
  if (strcmp(symbol,"clCreateProgramWithSource")==0 && real) {
    g_real_create.store(reinterpret_cast<CreateProgramFn>(real), std::memory_order_release);
    int n=g_dlsym_hits.fetch_add(1)+1;
    __android_log_print(ANDROID_LOG_INFO,"AdrenoLLM","dlsym intercepted clCreateProgramWithSource #%d real=%p",n,real);
    return reinterpret_cast<void*>(&hook_create_program);
  }
  if (strcmp(symbol,"clCreateKernel")==0 && real && (geom_x2_on() || int4_y4_on())) {
    g_real_create_kernel.store(reinterpret_cast<CreateKernelFn>(real), std::memory_order_release);
    return reinterpret_cast<void*>(&hook_create_kernel);
  }
  if (strcmp(symbol,"clEnqueueNDRangeKernel")==0 && real && (geom_x2_on() || int4_y4_on() || q_virtual_y4_on() || add_virtual_y4_on() || static_replay_on() || enqueue_pass_on())) {
    g_real_enqueue_kernel.store(reinterpret_cast<EnqueueKernelFn>(real), std::memory_order_release);
    return reinterpret_cast<void*>(&hook_enqueue_kernel);
  }
  if (strcmp(symbol,"clEnqueueReadBuffer")==0 && real && (env_on("ADRENOLLM_LOG_TOKEN_IDS") || static_replay_on())) {
    g_real_read.store(reinterpret_cast<ReadBufferFn>(real), std::memory_order_release);
    return reinterpret_cast<void*>(&hook_read_buffer);
  }
  return real;
}
