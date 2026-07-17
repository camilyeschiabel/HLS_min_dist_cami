#include "bitnet.h"
#include <math.h> // Fallback to math.h if hls_math.h cannot be loaded in simulation

#ifdef __SYNTHESIS__
#include <hls_math.h>
#define SQRT hls::sqrt
#define EXP hls::exp
#define COS hls::cos
#define SIN hls::sin
#define POW hls::pow
#else
#define SQRT sqrtf
#define EXP expf
#define COS cosf
#define SIN sinf
#define POW powf

// ---------------------------------------------------------------------------
// Tabela de decodificação de 2 bits para peso ternário:
//   índice 0 (00) -> 0  |  1 (01) -> +1  |  2 (10) -> -1  |  3 (11) -> 0
// ---------------------------------------------------------------------------
static const signed char DECODE_TABLE[4] = {0, 1, -1, 0};

void unpack_weights(uint8_t packed, weight_t *w0, weight_t *w1,
                    weight_t *w2, weight_t *w3) {
  *w0 = DECODE_TABLE[(packed     ) & 0x3];
  *w1 = DECODE_TABLE[(packed >> 2) & 0x3];
  *w2 = DECODE_TABLE[(packed >> 4) & 0x3];
  *w3 = DECODE_TABLE[(packed >> 6) & 0x3];
}
#endif

// ----------------------------------------------------------------------------
// Dump de Ativações
// ----------------------------------------------------------------------------

#ifdef DUMP_ACTIVATIONS
#include <stdio.h>
#include <sys/stat.h>

static char dump_dir[256] = "dump/token_0";

void set_dump_dir(const char *dir) {
    snprintf(dump_dir, sizeof(dump_dir), "%s", dir);
}

static void dump_array(const char *name, const float *data, int size) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), size, f);
        fclose(f);
    }
}
#endif

void bitlinear(const act_t *in, const weight_t *W, act_t *out, int in_features,
               int out_features, act_t scale) {
  // 1. Activation Quantization (equivalent to s = 127 / max(abs(x)))
  float max_abs = 1e-5f;
  for (int j = 0; j < in_features; j++) {
    float abs_x = in[j] > 0 ? in[j] : -in[j];
    if (abs_x > max_abs)
      max_abs = abs_x;
  }
  float s = 127.0f / max_abs;
  float inv_s = 1.0f / s;

#if BITNET_VERSION == 0
  // V0: Quantização e if/else de pesos feitos dentro do loop de multiplicação
  for (int i = 0; i < out_features; i++) {
    accum_t acc = 0;
    for (int j = 0; j < in_features; j++) {
      float scaled_in = in[j] * s;
      scaled_in = scaled_in > 0 ? (int)(scaled_in + 0.5f) : (int)(scaled_in - 0.5f);
      if (scaled_in > 127.0f)  scaled_in = 127.0f;
      if (scaled_in < -128.0f) scaled_in = -128.0f;

      weight_t w = W[i * in_features + j];
      if (w == 1)       acc += scaled_in;
      else if (w == -1) acc -= scaled_in;
    }
    out[i] = (act_t)(acc * inv_s) * scale;
  }
#else
  // V1 & V2: Pré-quantização feita uma vez fora do loop
  float quant_in[HIDDEN_DIM];
  for (int j = 0; j < in_features; j++) {
    float scaled_in = in[j] * s;
    scaled_in =
        scaled_in > 0 ? (int)(scaled_in + 0.5f) : (int)(scaled_in - 0.5f);
    if (scaled_in > 127.0f)
      scaled_in = 127.0f;
    if (scaled_in < -128.0f)
      scaled_in = -128.0f;
    quant_in[j] = scaled_in;
  }

  // 3. Matrix Multiplication
  #if BITNET_VERSION == 2
    #ifndef __SYNTHESIS__
      // Simulação C (V2): W é um ponteiro para bytes compactados (4 pesos ternários/byte).
      // Descompactamos 4 pesos por vez, eliminando 4x os acessos à RAM.
      const uint8_t *W_packed = (const uint8_t *)W;
      for (int i = 0; i < out_features; i++) {
        accum_t acc = 0;
        // in_features é sempre múltiplo de 4 (2560, 6912, 640) — loop sem resto
        for (int j = 0; j < in_features; j += 4) {
          uint8_t packed = W_packed[(i * in_features + j) / 4];
          weight_t w0, w1, w2, w3;
          unpack_weights(packed, &w0, &w1, &w2, &w3);
          acc += quant_in[j    ] * w0;
          acc += quant_in[j + 1] * w1;
          acc += quant_in[j + 2] * w2;
          acc += quant_in[j + 3] * w3;
        }
        // Dequantize output
        out[i] = (act_t)(acc * inv_s) * scale;
      }
    #else
      // Síntese HLS (V2): W é ap_int<2>, leitura direta — sem mudança alguma.
      for (int i = 0; i < out_features; i++) {
        accum_t acc = 0;
        for (int j = 0; j < in_features; j++) {
          weight_t w = W[i * in_features + j];
          acc += quant_in[j] * w;
        }
        // Dequantize output
        out[i] = (act_t)(acc * inv_s) * scale;
      }
    #endif
  #else
    // V1: Matrix Multiplication — produto branchless com pesos normais
    for (int i = 0; i < out_features; i++) {
      accum_t acc = 0;
      for (int j = 0; j < in_features; j++) {
        weight_t w = W[i * in_features + j];
        acc += quant_in[j] * w;
      }
      // Dequantize output
      out[i] = (act_t)(acc * inv_s) * scale;
    }
  #endif
#endif
}

void rms_norm(const act_t *in, const act_t *weight, act_t *out, int size,
              float eps) {
  accum_t ss = 0;
  for (int i = 0; i < size; i++) {
    ss += in[i] * in[i];
  }
  ss /= size;
  ss += eps;
  act_t inv_rms = 1.0f / SQRT((float)ss);

  for (int i = 0; i < size; i++) {
    out[i] = in[i] * inv_rms * weight[i];
  }
}

void squared_relu_glu(const act_t *gate, const act_t *up, act_t *out,
                      int size) {
  // F.relu(x)**2 * up
  for (int i = 0; i < size; i++) {
    act_t x = gate[i];
    act_t relu2 = (x > 0.0f) ? (x * x) : 0.0f;
    out[i] = relu2 * up[i];
  }
}

void rope(act_t *q, act_t *k, int seq_pos, int head_dim, int num_heads) {
  // Q
  for (int h = 0; h < num_heads; h++) {
    for (int d = 0; d < head_dim; d += 2) {
      float theta = seq_pos * POW(500000.0f, -((float)d) / head_dim);
      float cos_theta = COS(theta);
      float sin_theta = SIN(theta);

      int idx = h * head_dim + d;

      float q0 = q[idx];
      float q1 = q[idx + 1];
      q[idx] = q0 * cos_theta - q1 * sin_theta;
      q[idx + 1] = q0 * sin_theta + q1 * cos_theta;
    }
  }
  // K (GQA isolated)
  for (int h = 0; h < NUM_KV_HEADS; h++) {
    for (int d = 0; d < head_dim; d += 2) {
      float theta = seq_pos * POW(500000.0f, -((float)d) / head_dim);
      float cos_theta = COS(theta);
      float sin_theta = SIN(theta);

      int idx = h * head_dim + d;

      float k0 = k[idx];
      float k1 = k[idx + 1];
      k[idx] = k0 * cos_theta - k1 * sin_theta;
      k[idx + 1] = k0 * sin_theta + k1 * cos_theta;
    }
  }
}

static act_t k_cache[NUM_LAYERS][MAX_SEQ_LEN][NUM_KV_HEADS * HEAD_DIM];
static act_t v_cache[NUM_LAYERS][MAX_SEQ_LEN][NUM_KV_HEADS * HEAD_DIM];

void attention(const act_t *q, const act_t *k, const act_t *v, act_t *out,
               int layer, int seq_pos, int num_heads, int head_dim) {
  int num_kv_groups = num_heads / NUM_KV_HEADS;
  float scale = 1.0f / SQRT((float)head_dim);

  // Wait for compiler optimization memory barriers if necessary.
  // Sequentially cache KV vectors natively
  for (int d = 0; d < NUM_KV_HEADS * head_dim; d++) {
    k_cache[layer][seq_pos][d] = k[d];
    v_cache[layer][seq_pos][d] = v[d];
  }

  act_t scores[MAX_SEQ_LEN];
  for (int h = 0; h < num_heads; h++) {
    int kv_h = h / num_kv_groups;

    // Exact Math: Q * K.T * scale (scaled dot-product)
    for (int p = 0; p <= seq_pos; p++) {
      float score = 0;
      for (int d = 0; d < head_dim; d++) {
        score += q[h * head_dim + d] * k_cache[layer][p][kv_h * head_dim + d];
      }
      scores[p] = score * scale;
    }

    // Softmax
    float max_score = -1e9f;
    for (int p = 0; p <= seq_pos; p++) {
      if (scores[p] > max_score)
        max_score = scores[p];
    }
    float sum_exp = 0;
    for (int p = 0; p <= seq_pos; p++) {
      scores[p] = EXP(scores[p] - max_score);
      sum_exp += scores[p];
    }
    for (int p = 0; p <= seq_pos; p++) {
      scores[p] /= sum_exp;
    }

    // Exact Math: Score * V (Weighted sums)
    for (int d = 0; d < head_dim; d++) {
      float val = 0;
      for (int p = 0; p <= seq_pos; p++) {
        val += scores[p] * v_cache[layer][p][kv_h * head_dim + d];
      }
      out[h * head_dim + d] = val;
    }
  }
}

void transformer_layer(
    const act_t *in, act_t *out, const weight_t *q_proj_w,
    const weight_t *k_proj_w, const weight_t *v_proj_w,
    const weight_t *o_proj_w, const weight_t *gate_proj_w,
    const weight_t *up_proj_w, const weight_t *down_proj_w,
    const act_t *attn_norm_w, const act_t *attn_sub_norm_w,
    const act_t *ffn_norm_w, const act_t *ffn_sub_norm_w,
    const act_t *layer_scales, /* 7 scales per layer for the 7 bitlinears */
    int layer, int seq_pos, float rms_eps
#ifdef DUMP_ACTIVATIONS
    , int do_dump
#endif
    ) {
  act_t norm_out[EMBED_DIM];
  act_t q[EMBED_DIM];
  act_t k[EMBED_DIM];
  act_t v[EMBED_DIM];
  act_t attn_out[EMBED_DIM];
  act_t o_out[EMBED_DIM];

  int kv_dim = NUM_KV_HEADS * HEAD_DIM;

#ifdef DUMP_ACTIVATIONS
  char name_buf[128];
  #define DNAME(fmt) (snprintf(name_buf, sizeof(name_buf), fmt, layer), name_buf)
#endif

#ifdef TIMING_BREAKDOWN
  timing_layer = layer;
#endif

  // 1. Attention Block
  TIME_COMPONENT("rms_norm_attn",
      rms_norm(in, attn_norm_w, norm_out, EMBED_DIM, rms_eps));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("rms_norm_layer%02d_attn"), norm_out, EMBED_DIM);
#endif

  TIME_COMPONENT("bitlinear_Q",
      bitlinear(norm_out, q_proj_w, q, EMBED_DIM, EMBED_DIM, layer_scales[0]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_Q"), q, EMBED_DIM);
#endif

  TIME_COMPONENT("bitlinear_K",
      bitlinear(norm_out, k_proj_w, k, EMBED_DIM, kv_dim, layer_scales[1]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_K"), k, kv_dim);
#endif

  TIME_COMPONENT("bitlinear_V",
      bitlinear(norm_out, v_proj_w, v, EMBED_DIM, kv_dim, layer_scales[2]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_V"), v, kv_dim);
#endif

  double t0 = get_time_in_seconds();
  TIME_COMPONENT("rope", rope(q, k, seq_pos, HEAD_DIM, NUM_HEADS));
  total_rope_time += get_time_in_seconds() - t0;
  TIME_COMPONENT("attention",
      attention(q, k, v, attn_out, layer, seq_pos, NUM_HEADS, HEAD_DIM));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("attention_layer%02d"), attn_out, EMBED_DIM);
#endif

  // Sub-norm natively scales attention output before projection
  act_t attn_sub_out[EMBED_DIM];
  TIME_COMPONENT("rms_norm_attn_sub",
      rms_norm(attn_out, attn_sub_norm_w, attn_sub_out, EMBED_DIM, rms_eps));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("rms_norm_layer%02d_attn_sub"), attn_sub_out, EMBED_DIM);
#endif

  TIME_COMPONENT("bitlinear_O",
      bitlinear(attn_sub_out, o_proj_w, o_out, EMBED_DIM, EMBED_DIM,
                layer_scales[3]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_O"), o_out, EMBED_DIM);
#endif

  // Residual Add
  act_t ffn_in[EMBED_DIM];
  for (int i = 0; i < EMBED_DIM; i++)
    ffn_in[i] = in[i] + o_out[i];

  // 2. FFN Block
  act_t gate[HIDDEN_DIM];
  act_t up[HIDDEN_DIM];
  act_t swi[HIDDEN_DIM];
  act_t down[EMBED_DIM];

  TIME_COMPONENT("rms_norm_ffn",
      rms_norm(ffn_in, ffn_norm_w, norm_out, EMBED_DIM, rms_eps));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("rms_norm_layer%02d_ffn"), norm_out, EMBED_DIM);
#endif

  TIME_COMPONENT("bitlinear_Gate",
      bitlinear(norm_out, gate_proj_w, gate, EMBED_DIM, HIDDEN_DIM,
                layer_scales[4]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_Gate"), gate, HIDDEN_DIM);
#endif

  TIME_COMPONENT("bitlinear_Up",
      bitlinear(norm_out, up_proj_w, up, EMBED_DIM, HIDDEN_DIM, layer_scales[5]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_Up"), up, HIDDEN_DIM);
#endif

  TIME_COMPONENT("squared_relu_glu",
      squared_relu_glu(gate, up, swi, HIDDEN_DIM));

  // Sub-norm natively scales SwiGLU activation before projection
  act_t ffn_sub_out[HIDDEN_DIM];
  TIME_COMPONENT("rms_norm_ffn_sub",
      rms_norm(swi, ffn_sub_norm_w, ffn_sub_out, HIDDEN_DIM, rms_eps));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("rms_norm_layer%02d_ffn_sub"), ffn_sub_out, HIDDEN_DIM);
#endif

  TIME_COMPONENT("bitlinear_Down",
      bitlinear(ffn_sub_out, down_proj_w, down, HIDDEN_DIM, EMBED_DIM,
                layer_scales[6]));
#ifdef DUMP_ACTIVATIONS
  if (do_dump) dump_array(DNAME("bitlinear_layer%02d_Down"), down, EMBED_DIM);
#endif

  // Final Residual Add
  for (int i = 0; i < EMBED_DIM; i++)
    out[i] = ffn_in[i] + down[i];
}

void dense_linear(const act_t *in, const act_t *W, act_t *out, int in_features,
                  int out_features) {
  for (int i = 0; i < out_features; i++) {
    accum_t acc = 0;
    for (int j = 0; j < in_features; j++) {
      acc += in[j] * W[i * in_features + j];
    }
    out[i] = (act_t)acc;
  }
}

void bitnet_inference(const act_t *token_embed, act_t *logits,
                      const weight_t *all_layer_weights,
                      const act_t *all_layer_norms,
                      const act_t *all_layer_scales, const act_t *lm_head_w,
                      const act_t *final_norm_w, int seq_pos
#ifdef DUMP_ACTIVATIONS
                      , int dump_this_token
                      , const char *token_label
#endif
                      ) {
  act_t ping[EMBED_DIM];
  act_t pong[EMBED_DIM];

  for (int i = 0; i < EMBED_DIM; i++)
    ping[i] = token_embed[i];

  long long w_offset = 0;
  long long norm_offset = 0;

  const long long W_Q_SIZE = (long long)EMBED_DIM * EMBED_DIM;
  const long long W_KV_SIZE = (long long)EMBED_DIM * (NUM_KV_HEADS * HEAD_DIM);
  const long long W_MLP_SIZE = (long long)EMBED_DIM * HIDDEN_DIM;
  const float RMS_EPS = 1e-5f;

#ifndef __SYNTHESIS__
  #if BITNET_VERSION == 2
    #define W_PTR(offset) &all_layer_weights[(offset) / 4]
  #else
    #define W_PTR(offset) &all_layer_weights[(offset)]
  #endif
#else
  #define W_PTR(offset) &all_layer_weights[(offset)]
#endif

  for (int l = 0; l < NUM_LAYERS; l++) {
    const weight_t *q_w = W_PTR(w_offset);
    w_offset += W_Q_SIZE;
    const weight_t *k_w = W_PTR(w_offset);
    w_offset += W_KV_SIZE;
    const weight_t *v_w = W_PTR(w_offset);
    w_offset += W_KV_SIZE;
    const weight_t *o_w = W_PTR(w_offset);
    w_offset += W_Q_SIZE;

    const weight_t *gate_w = W_PTR(w_offset);
    w_offset += W_MLP_SIZE;
    const weight_t *up_w = W_PTR(w_offset);
    w_offset += W_MLP_SIZE;
    const weight_t *down_w = W_PTR(w_offset);
    w_offset += W_MLP_SIZE;

    const act_t *attn_n = &all_layer_norms[norm_offset];
    norm_offset += EMBED_DIM;
    const act_t *attn_sub_n = &all_layer_norms[norm_offset];
    norm_offset += EMBED_DIM;
    const act_t *ffn_n = &all_layer_norms[norm_offset];
    norm_offset += EMBED_DIM;
    const act_t *ffn_sub_n = &all_layer_norms[norm_offset];
    norm_offset += HIDDEN_DIM;

    const act_t *l_scales = &all_layer_scales[l * 7];

    act_t *in_buf = (l % 2 == 0) ? ping : pong;
    act_t *out_buf = (l % 2 == 0) ? pong : ping;

#ifdef DUMP_ACTIVATIONS
    if (dump_this_token) set_dump_dir(token_label);
#endif

    transformer_layer(in_buf, out_buf, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
                      attn_n, attn_sub_n, ffn_n, ffn_sub_n, l_scales, l,
                      seq_pos, RMS_EPS
#ifdef DUMP_ACTIVATIONS
                      , dump_this_token
#endif
                      );
  }

  act_t *final_hidden = (NUM_LAYERS % 2 == 0) ? ping : pong;
  act_t final_norm_out[EMBED_DIM];

#ifdef TIMING_BREAKDOWN
  timing_layer = -1;  // fora das camadas (final_norm / lm_head)
#endif

  TIME_COMPONENT("rms_norm_final",
      rms_norm(final_hidden, final_norm_w, final_norm_out, EMBED_DIM, RMS_EPS));

#ifdef DUMP_ACTIVATIONS
  if (dump_this_token)
      dump_array("final_norm", final_norm_out, EMBED_DIM);
#endif

  double t1 = get_time_in_seconds();
  TIME_COMPONENT("lm_head",
      dense_linear(final_norm_out, lm_head_w, logits, EMBED_DIM, VOCAB_SIZE));
  total_lm_head_time += get_time_in_seconds() - t1;

#ifdef DUMP_ACTIVATIONS
  if (dump_this_token) {
      char logits_path[512];
      snprintf(logits_path, sizeof(logits_path),
               "%s/logits.bin", token_label);
      FILE *f = fopen(logits_path, "wb");
      if (f) {
          fwrite(logits, sizeof(float), VOCAB_SIZE, f);
          fclose(f);
      }
  }
#endif
}
