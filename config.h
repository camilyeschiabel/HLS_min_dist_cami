#ifndef CONFIG_H
#define CONFIG_H

// Define a versão ativa do BitNet:
// 0 = V0 (Original ingênua, quantização no loop, if/else para pesos)
// 1 = V1 (Pré-quantização antes do loop, multiplicação branchless)
// 2 = V2 (Descompactação de pesos compactados de 2 bits)
#ifndef BITNET_VERSION
#define BITNET_VERSION 1
#endif

// ============================================================================
// BitNet b1.58 (2B-4T) HLS configuration configuration
// ============================================================================

// BitNet b1.58 2B-4T Hyperparameters
#define EMBED_DIM   2560
#define HIDDEN_DIM  6912
#define NUM_HEADS   20
#define NUM_KV_HEADS 5
#define NUM_LAYERS  30

// The dimension per attention head (usually EMBED_DIM / NUM_HEADS = 80 or 128 depending on GQA)
#define HEAD_DIM    (EMBED_DIM / NUM_HEADS)

// The total vocabulary size for the output LM head.
#define VOCAB_SIZE  128256

// Max sequence length supported by the hardware constraints
#define MAX_SEQ_LEN 2048

// Hardware Data Types
// ----------------------------------------------------------------------------
#ifdef __SYNTHESIS__
#include <ap_int.h>
typedef ap_int<2> weight_t;
#else
// Mock for PC simulation without Vitis HLS environment loaded
typedef signed char weight_t;
#endif

// act_t: Represents the activations. Standard float is used to maintain precision
// across LayerNorms, RoPE, and non-linearities. Alternatively, half-precision 
// could be used, but float ensures numerical stability matching the reference.
typedef float act_t;

// accum_t: High-precision accumulator type. Used during the innermost loops of
// matrix multiplications (even ternary ones) to prevent overflow before quantization 
// or normalization.
typedef float accum_t;

#endif // CONFIG_H