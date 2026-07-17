#include "bitnet.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

double total_lm_head_time = 0.0;
double total_rope_time    = 0.0;

#ifdef TIMING_BREAKDOWN
FILE *timing_csv       = NULL;
int   timing_phase     = 0;
int   timing_token_idx = 0;
int   timing_layer     = 0;

static void init_timing_csv(const char *path) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        mkdir(dir, 0755);
    }
    timing_csv = fopen(path, "w");
    if (timing_csv) {
        fprintf(timing_csv, "fase,token_idx,layer,componente,tempo_s\n");
        printf("[TIMING] CSV aberto: %s\n", path);
    } else {
        printf("[TIMING] Erro ao abrir CSV: %s\n", path);
    }
}
#endif

double get_time_in_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ----------------------------------------------------------------------------
// Dump de Ativações
// ----------------------------------------------------------------------------

#ifdef DUMP_ACTIVATIONS
#include <sys/stat.h>

typedef struct {
    int  first_idx;
    int  prefill_idx;
    int  generated_idx;
    char dir_first[512];
    char dir_prefill[512];
    char dir_generated[512];
    int  valid;
} DumpConfig;

static DumpConfig dump_cfg = {0};

static void load_dump_config(void) {
    const char *v;

    v = getenv("DUMP_FIRST_IDX");
    dump_cfg.first_idx = (v && v[0]) ? atoi(v) : 0;

    v = getenv("DUMP_PREFILL_IDX");
    dump_cfg.prefill_idx = (v && v[0]) ? atoi(v) : -1;

    v = getenv("DUMP_GENERATED_IDX");
    dump_cfg.generated_idx = (v && v[0]) ? atoi(v) : -1;

    v = getenv("DUMP_DIR_FIRST");
    if (v && v[0])
        strncpy(dump_cfg.dir_first, v, sizeof(dump_cfg.dir_first) - 1);

    v = getenv("DUMP_DIR_PREFILL");
    if (v && v[0])
        strncpy(dump_cfg.dir_prefill, v, sizeof(dump_cfg.dir_prefill) - 1);

    v = getenv("DUMP_DIR_GENERATED");
    if (v && v[0])
        strncpy(dump_cfg.dir_generated, v, sizeof(dump_cfg.dir_generated) - 1);

    dump_cfg.valid = 1;

    printf("[DUMP] Configuração carregada:\n");
    printf("  first_idx=%d     dir=%s\n", dump_cfg.first_idx,
           dump_cfg.dir_first);
    printf("  prefill_idx=%d   dir=%s\n", dump_cfg.prefill_idx,
           dump_cfg.dir_prefill);
    printf("  generated_idx=%d dir=%s\n", dump_cfg.generated_idx,
           dump_cfg.dir_generated);
}

static int should_dump(int seq_pos, char *label_out, int label_size) {
    if (!dump_cfg.valid) return 0;

    if (seq_pos == dump_cfg.first_idx && dump_cfg.dir_first[0]) {
        strncpy(label_out, dump_cfg.dir_first, label_size - 1);
        return 1;
    }
    if (seq_pos == dump_cfg.prefill_idx && dump_cfg.dir_prefill[0]) {
        strncpy(label_out, dump_cfg.dir_prefill, label_size - 1);
        return 1;
    }
    if (seq_pos == dump_cfg.generated_idx && dump_cfg.dir_generated[0]) {
        strncpy(label_out, dump_cfg.dir_generated, label_size - 1);
        return 1;
    }
    return 0;
}
#endif

// ----------------------------------------------------------------------------
// Embedding lookup
// ----------------------------------------------------------------------------

void get_embedding(int token_id, act_t *embed_out,
                   const act_t *embeddings_table) {
    if (token_id < 0 || token_id >= VOCAB_SIZE)
        token_id = 0;
    for (int i = 0; i < EMBED_DIM; i++)
        embed_out[i] = embeddings_table[token_id * EMBED_DIM + i];
}

// ----------------------------------------------------------------------------
// Wrapper de inferência
// ----------------------------------------------------------------------------

static void run_inference(const act_t *token_embed, act_t *logits,
                          const weight_t *all_weights, const act_t *all_norms,
                          const act_t *all_scales, const act_t *embeddings_table,
                          const act_t *final_norm, int seq_pos) {
#ifdef DUMP_ACTIVATIONS
    char label[512] = {0};
    int  do_dump    = should_dump(seq_pos, label, sizeof(label));
    if (do_dump)
        printf("[DUMP] Capturando seq_pos=%d → %s\n", seq_pos, label);
    bitnet_inference(token_embed, logits, all_weights, all_norms, all_scales,
                     embeddings_table, final_norm, seq_pos, do_dump, label);
#else
    bitnet_inference(token_embed, logits, all_weights, all_norms, all_scales,
                     embeddings_table, final_norm, seq_pos);
#endif
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    printf("==========================================\n");
    printf(" BitNet b1.58 (2B-4T) HLS Simulation\n");
    printf("==========================================\n\n");

#ifdef DUMP_ACTIVATIONS
    load_dump_config();
    printf("\n");
#endif

#ifdef TIMING_BREAKDOWN
    const char *timing_path = getenv("TIMING_CSV_PATH");
    if (!timing_path || timing_path[0] == '\0')
        timing_path = "timing_breakdown.csv";
    init_timing_csv(timing_path);
    printf("\n");
#endif

    double total_embed_time   = 0.0;
    double total_prefill_time = 0.0;
    struct timespec start_time, end_time;
    struct timespec prog_start, prog_end;
    struct timespec prefill_start, prefill_end;

    // 1. Alocação de memória
    long long embed_size = (long long)VOCAB_SIZE * EMBED_DIM;
    long long kv_dim     = NUM_KV_HEADS * HEAD_DIM;

    long long layer_weights =
        (long long)EMBED_DIM * EMBED_DIM      // Q
      + (long long)EMBED_DIM * kv_dim         // K
      + (long long)EMBED_DIM * kv_dim         // V
      + (long long)EMBED_DIM * EMBED_DIM      // O
      + (long long)EMBED_DIM * HIDDEN_DIM     // Gate
      + (long long)EMBED_DIM * HIDDEN_DIM     // Up
      + (long long)HIDDEN_DIM * EMBED_DIM;    // Down

    long long total_weights        = (long long)NUM_LAYERS * layer_weights;
    long long total_weights_packed = total_weights / 4;
    long long num_norm_elems       = (long long)NUM_LAYERS * (3 * EMBED_DIM + HIDDEN_DIM);
    long long num_scale_elems      = (long long)NUM_LAYERS * 7;

    // Determinar arquivo de pesos e tamanho de alocação
    const char *weights_file = getenv("WEIGHTS_FILE");
    if (!weights_file || weights_file[0] == '\0') {
#if BITNET_VERSION == 2
        weights_file = "weights_packed.bin";
#else
        weights_file = "weights.bin";
#endif
    }

    int       is_packed  = (strstr(weights_file, "packed") != NULL);
    long long alloc_size = is_packed ? total_weights_packed : total_weights;

#if BITNET_VERSION == 2
    if (!is_packed) {
        printf("[WARNING] BITNET_VERSION == 2 (Packed weights) mas o arquivo de pesos '%s' NAO parece compactado!\n", weights_file);
    }
#else
    if (is_packed) {
        printf("[WARNING] BITNET_VERSION == %d (Pesos descompactados) mas o arquivo de pesos '%s' parece compactado!\n", BITNET_VERSION, weights_file);
    }
#endif

    act_t   *embeddings_table = (act_t *)  malloc(embed_size        * sizeof(act_t));
    uint8_t *all_weights      = (uint8_t *)malloc(alloc_size);
    act_t   *all_norms        = (act_t *)  malloc(num_norm_elems     * sizeof(act_t));
    act_t   *all_scales       = (act_t *)  malloc(num_scale_elems    * sizeof(act_t));
    act_t   *final_norm       = (act_t *)  malloc(EMBED_DIM          * sizeof(act_t));

    if (!all_weights || !embeddings_table || !all_norms ||
        !all_scales  || !final_norm) {
        printf("Falha ao alocar memória.\n");
        return 1;
    }

    // Carregamento de norms.bin
    FILE *nf = fopen("../../../../norms.bin", "rb");
    if (!nf) nf = fopen("norms.bin", "rb");
    if (nf) {
        printf("Carregando norms.bin...\n");
        fread(all_norms,  sizeof(act_t), num_norm_elems, nf);
        fread(final_norm, sizeof(act_t), EMBED_DIM,      nf);
        fclose(nf);
    } else {
        printf("norms.bin não encontrado. Usando 1.0f.\n");
        for (long long i = 0; i < num_norm_elems; i++) all_norms[i]  = 1.0f;
        for (int i = 0; i < EMBED_DIM; i++)             final_norm[i] = 1.0f;
    }

    // Carregamento de scales.bin
    FILE *sf = fopen("../../../../scales.bin", "rb");
    if (!sf) sf = fopen("scales.bin", "rb");
    if (sf) {
        printf("Carregando scales.bin...\n");
        fread(all_scales, sizeof(act_t), num_scale_elems, sf);
        fclose(sf);
    } else {
        printf("scales.bin não encontrado. Usando 1.0f.\n");
        for (long long i = 0; i < num_scale_elems; i++) all_scales[i] = 1.0f;
    }

    // Carregamento de embeddings.bin
    FILE *ef = fopen("../../../../embeddings.bin", "rb");
    if (!ef) ef = fopen("embeddings.bin", "rb");
    if (ef) {
        printf("Carregando embeddings.bin...\n");
        fread(embeddings_table, sizeof(act_t), embed_size, ef);
        fclose(ef);
    } else {
        printf("embeddings.bin não encontrado. Usando 0.01f.\n");
        for (long long i = 0; i < embed_size; i++) embeddings_table[i] = 0.01f;
    }

    // Carregamento dos pesos
    char weights_path_alt[512];
    snprintf(weights_path_alt, sizeof(weights_path_alt),
             "../../../../%s", weights_file);

    FILE *wf = fopen(weights_path_alt, "rb");
    if (!wf) wf = fopen(weights_file, "rb");
    if (wf) {
        printf("Carregando %s (%s)...\n",
               weights_file,
               is_packed ? "4 pesos/byte" : "1 peso/byte");
        fread(all_weights, 1, alloc_size, wf);
        fclose(wf);
        printf("Pesos carregados.\n");
    } else {
        printf("%s não encontrado. Usando dados dummy.\n", weights_file);
        if (is_packed) {
            static const uint8_t DUMMY_PATTERN[3] = {0x92, 0x24, 0x49};
            for (long long i = 0; i < alloc_size; i++)
                all_weights[i] = DUMMY_PATTERN[i % 3];
        } else {
            for (long long i = 0; i < alloc_size; i++)
                all_weights[i] = (uint8_t)((i % 3) - 1);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &prog_start);

    // 2. Processamento do prompt
    char prompt_buffer[1024];
    prompt_buffer[0] = '\0';
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            strncat(prompt_buffer, argv[i],
                    sizeof(prompt_buffer) - strlen(prompt_buffer) - 1);
            if (i < argc - 1)
                strncat(prompt_buffer, " ",
                        sizeof(prompt_buffer) - strlen(prompt_buffer) - 1);
        }
    } else {
        strncpy(prompt_buffer, "128000", sizeof(prompt_buffer) - 1);
    }
    prompt_buffer[1023] = '\0';

    printf("\nTokens de entrada: '%s'\n\n", prompt_buffer);
    printf("Iniciando geração autorregressiva:\n");

    int    seq_pos      = 0;
#ifdef TIMING_BREAKDOWN
    int    global_token_idx = 0;
#endif
    act_t  token_embed[EMBED_DIM];
    act_t *logits = (act_t *)malloc(VOCAB_SIZE * sizeof(act_t));

    // --- Fase de Prefill ---
    clock_gettime(CLOCK_MONOTONIC, &prefill_start);
    char *word          = strtok(prompt_buffer, " \t\n");
    int   current_token = 0;

    while (word != NULL) {
        current_token = atoi(word);
        printf("Pre-fill Token: %d  (seq_pos=%d)\n", current_token, seq_pos);

#ifdef TIMING_BREAKDOWN
        timing_phase     = 0;  // prefill
        timing_token_idx = global_token_idx;
#endif

        clock_gettime(CLOCK_MONOTONIC, &start_time);
        get_embedding(current_token, token_embed, embeddings_table);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        total_embed_time += (end_time.tv_sec - start_time.tv_sec)
                          + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

        run_inference(token_embed, logits,
                      (const weight_t *)all_weights,
                      all_norms, all_scales,
                      embeddings_table, final_norm, seq_pos);
        seq_pos++;
#ifdef TIMING_BREAKDOWN
        global_token_idx++;
#endif
        word = strtok(NULL, " \t\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &prefill_end);
    total_prefill_time = (prefill_end.tv_sec - prefill_start.tv_sec)
                       + (prefill_end.tv_nsec - prefill_start.tv_nsec) / 1e9;

    // --- Fase de Decode ---
    int max_gen = 10;
    const char *max_gen_env = getenv("MAX_GEN");
    if (max_gen_env && max_gen_env[0]) max_gen = atoi(max_gen_env);

    double total_decode_time       = 0.0;
    double total_post_process_time = 0.0;
    struct timespec decode_start, decode_end;
    clock_gettime(CLOCK_MONOTONIC, &decode_start);

    for (int step = 0; step < max_gen; step++) {
        struct timespec post_start, post_end;
        clock_gettime(CLOCK_MONOTONIC, &post_start);

        int   best_token = 0;
        float best_val   = -1e9f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (logits[i] > best_val) {
                best_val   = logits[i];
                best_token = i;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &post_end);
        total_post_process_time += (post_end.tv_sec - post_start.tv_sec)
                                 + (post_end.tv_nsec - post_start.tv_nsec) / 1e9;

        printf("[OUT]%d\n", best_token);
        fflush(stdout);

        current_token = best_token;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        get_embedding(current_token, token_embed, embeddings_table);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        total_embed_time += (end_time.tv_sec - start_time.tv_sec)
                          + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

#ifdef TIMING_BREAKDOWN
        timing_phase     = 1;  // decode
        timing_token_idx = global_token_idx;
#endif

        run_inference(token_embed, logits,
                      (const weight_t *)all_weights,
                      all_norms, all_scales,
                      embeddings_table, final_norm, seq_pos);
        seq_pos++;
#ifdef TIMING_BREAKDOWN
        global_token_idx++;
#endif
    }

    clock_gettime(CLOCK_MONOTONIC, &decode_end);
    total_decode_time = (decode_end.tv_sec - decode_start.tv_sec)
                      + (decode_end.tv_nsec - decode_start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &prog_end);
    double total_prog_time = (prog_end.tv_sec - prog_start.tv_sec)
                           + (prog_end.tv_nsec - prog_start.tv_nsec) / 1e9;

    // --- Relatório ---
    printf("\n\n[Simulation Complete]\n");
    printf("Tempo total de execucao (inferencia): %.6f segundos\n",
           total_prog_time);

    printf("\n[1. Input Embeddings (Tabela)]\n");
    printf("Tempo gasto: %.9f segundos\n", total_embed_time);
    if (total_prog_time > 0)
        printf("Porcentagem do tempo: %.6f%%\n",
               (total_embed_time / total_prog_time) * 100.0);

    printf("\n[2. Output Embeddings (LM Head)]\n");
    printf("Tempo gasto: %.6f segundos\n", total_lm_head_time);
    if (total_prog_time > 0)
        printf("Porcentagem do tempo: %.6f%%\n",
               (total_lm_head_time / total_prog_time) * 100.0);

    printf("\n[3. Rotary Positional Embeddings (RoPE)]\n");
    printf("Tempo gasto: %.6f segundos\n", total_rope_time);
    if (total_prog_time > 0)
        printf("Porcentagem do tempo: %.6f%%\n",
               (total_rope_time / total_prog_time) * 100.0);

    printf("\n[4. Prefill (Prompt Processing Phase)]\n");
    printf("Tempo gasto: %.6f segundos\n", total_prefill_time);
    if (total_prog_time > 0)
        printf("Porcentagem do tempo: %.6f%%\n",
               (total_prefill_time / total_prog_time) * 100.0);

    printf("\n[5. Decode (Generation Phase)]\n");
    printf("Tempo gasto: %.6f segundos\n", total_decode_time);
    if (total_prog_time > 0)
        printf("Porcentagem do tempo: %.6f%%\n",
               (total_decode_time / total_prog_time) * 100.0);

    printf("\n[6. Post-processing (Argmax)]\n");
    printf("Tempo gasto: %.6f segundos\n", total_post_process_time);
    if (total_prog_time > 0)
        printf("Porcentagem do tempo: %.6f%%\n",
               (total_post_process_time / total_prog_time) * 100.0);

    // Limpeza
    free(all_weights);
    free(all_norms);
    free(all_scales);
    free(final_norm);
    free(embeddings_table);
    free(logits);

#ifdef TIMING_BREAKDOWN
    if (timing_csv) {
        fclose(timing_csv);
        printf("\n[TIMING] CSV fechado.\n");
    }
#endif

    return 0;
}
