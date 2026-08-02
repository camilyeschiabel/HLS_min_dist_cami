#ifndef BITNET_H
#define BITNET_H
#include "config.h"

#ifdef __SYNTHESIS__
#define get_time_in_seconds() (0.0)
#define total_rope_time dummy_rope_time
#define total_lm_head_time dummy_lm_head_time
static double dummy_rope_time = 0.0;
static double dummy_lm_head_time = 0.0;
#else
#include <time.h>
#include <stdint.h>
extern double total_rope_time;
extern double total_lm_head_time;
extern unsigned long long g_bitlinear_calls;
extern unsigned long long g_bitlinear_elements;
double get_time_in_seconds(void);
#endif

// ----------------------------------------------------------------------------
// Timing Breakdown — CSV por componente/camada/token/fase
// ----------------------------------------------------------------------------
#ifdef TIMING_BREAKDOWN
#include <stdio.h>
extern FILE *timing_csv;
extern int   timing_phase;      // 0 = prefill, 1 = decode
extern int   timing_token_idx;  // índice global do token (prefill + decode)
extern int   timing_layer;      // -1 = fora das camadas (embed/lm_head/final_norm)

#define TIME_COMPONENT(name, code) do { \
    double _t0 = get_time_in_seconds(); \
    code; \
    double _t1 = get_time_in_seconds(); \
    if (timing_csv) { \
        fprintf(timing_csv, "%s,%d,%d,%s,%.9f\n", \
                (timing_phase == 0 ? "prefill" : "decode"), \
                timing_token_idx, timing_layer, name, _t1 - _t0); \
    } \
} while (0)
#else
#define TIME_COMPONENT(name, code) code
#endif

// ----------------------------------------------------------------------------
// Core Function Signatures
// ----------------------------------------------------------------------------

void bitlinear(const act_t *in, const weight_t *W, act_t *out, int in_features,
               int out_features, act_t scale);

void rms_norm(const act_t *in, const act_t *weight, act_t *out, int size,
              float eps);

void squared_relu_glu(const act_t *gate, const act_t *up, act_t *out, int size);

void rope(act_t *q, act_t *k, int seq_pos, int head_dim, int num_heads);

void transformer_layer(
    const act_t *in, act_t *out,
    const weight_t *q_proj_w, const weight_t *k_proj_w,
    const weight_t *v_proj_w, const weight_t *o_proj_w,
    const weight_t *gate_proj_w, const weight_t *up_proj_w,
    const weight_t *down_proj_w,
    const act_t *attn_norm_w, const act_t *attn_sub_norm_w,
    const act_t *ffn_norm_w,  const act_t *ffn_sub_norm_w,
    const act_t *layer_scales,
    int layer, int seq_pos, float rms_eps
#ifdef DUMP_ACTIVATIONS
    , int do_dump
#endif
    );

void bitnet_inference(const act_t *token_embed, act_t *logits,
                      const weight_t *all_layer_weights,
                      const act_t *all_layer_norms,
                      const act_t *all_layer_scales,
                      const act_t *lm_head_w,
                      const act_t *final_norm_w,
                      int seq_pos
#ifdef DUMP_ACTIVATIONS
                      , int dump_this_token
                      , const char *token_label
#endif
                      );

void attention(const act_t *q, const act_t *k, const act_t *v, act_t *out,
               int layer, int seq_pos, int num_heads, int head_dim);

void dense_linear(const act_t *in, const act_t *W, act_t *out,
                  int in_features, int out_features);

#endif // BITNET_H
