/*
 * skip-gram word2vec with negative sampling — fast C implementation
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -ffast-math -shared -fPIC -o train.so train.c -lm
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/* xorshift128+ PRNG                                                   */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t s[2];
} rng_t;

static inline uint64_t rng_next(rng_t *r) {
    uint64_t s1 = r->s[0];
    uint64_t s0 = r->s[1];
    r->s[0] = s0;
    s1 ^= s1 << 23;
    r->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return r->s[1] + s0;
}

static inline double rng_uniform(rng_t *r) {
    return (rng_next(r) >> 11) * 0x1.0p-53;
}

static inline int rng_int_n(rng_t *r, int n) {
    return (int)(rng_next(r) % (uint64_t)n);
}

static void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s[0] = z ^ (z >> 31);
    z += 0x9e3779b97f4a7c15ULL + 1;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    r->s[1] = z ^ (z >> 31);
}

/* ------------------------------------------------------------------ */
/* exp approximation via exp table                                     */
/* ------------------------------------------------------------------ */
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

static void build_exp_table(float *table) {
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        double x = (double)i / (double)EXP_TABLE_SIZE * 2.0 * MAX_EXP - MAX_EXP;
        table[i] = (float)(1.0 / (1.0 + exp(-x)));
    }
}

static inline float fast_sigmoid(float x, const float *exp_table) {
    if (x > MAX_EXP) return 1.0f;
    if (x < -MAX_EXP) return 0.0f;
    int idx = (int)((x + MAX_EXP) * (EXP_TABLE_SIZE / (2.0 * MAX_EXP)));
    if (idx < 0) idx = 0;
    if (idx >= EXP_TABLE_SIZE) idx = EXP_TABLE_SIZE - 1;
    return exp_table[idx];
}

/* ------------------------------------------------------------------ */
/* Unigram table for negative sampling (power 0.75)                     */
/* ------------------------------------------------------------------ */
static void build_unigram_table(const int *counts, int vocab_size,
                                int table_size, int *table) {
    double power = 0.75;
    double total = 0.0;
    double *cdf = (double *)malloc(vocab_size * sizeof(double));
    for (int i = 0; i < vocab_size; i++) {
        double p = pow((double)counts[i], power);
        cdf[i] = p;
        total += p;
    }
    int ti = 0;
    double acc = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        acc += cdf[i] / total;
        int slots = (int)(acc * table_size) - ti;
        if (slots < 0) slots = 0;
        for (int j = 0; j < slots && ti < table_size; j++, ti++)
            table[ti] = i;
    }
    while (ti < table_size) table[ti++] = 0;
    free(cdf);
}

/* ------------------------------------------------------------------ */
/* Subsampling probabilities                                           */
/* ------------------------------------------------------------------ */
static void build_subsample(const int *counts, int vocab_size,
                            long long total_words, double t,
                            float *prob) {
    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] == 0) { prob[i] = 1.0f; continue; }
        double f = (double)counts[i] / (double)total_words;
        double p = (sqrt(f / t) + 1.0) * (t / f);
        prob[i] = (float)((p > 1.0) ? 1.0 : p);
    }
}

/* ------------------------------------------------------------------ */
/* Main training entry point                                          */
/* ------------------------------------------------------------------ */
int train_skipgram(
    /* corpus: int array of word ids */
    const int *corpus, long long corpus_len,
    /* word counts (for unigram table and subsampling) */
    const int *counts, int vocab_size,
    /* input vectors: vocab_size x dim, row-major, init [-0.5/dim, 0.5/dim] */
    float *w_in,
    /* output vectors: vocab_size x dim, row-major, init zeros */
    float *w_out,
    int dim,
    /* sum of all word counts */
    long long total_words,
    int epochs, int window, int neg_samples,
    double subsample_t, float lr_start,
    int num_threads, uint64_t seed)
{
    /* Build data structures */
    const int UNIGRAM_TABLE_SIZE = 100000000;
    int *unigram_table = (int *)malloc(UNIGRAM_TABLE_SIZE * sizeof(int));
    if (!unigram_table) return -1;

    float *sub_prob = (float *)malloc(vocab_size * sizeof(float));
    if (!sub_prob) { free(unigram_table); return -1; }

    float exp_table[EXP_TABLE_SIZE];
    build_exp_table(exp_table);

    build_unigram_table(counts, vocab_size, UNIGRAM_TABLE_SIZE, unigram_table);
    build_subsample(counts, vocab_size, total_words, subsample_t, sub_prob);

    fprintf(stderr, "vocab=%d dim=%d corpus_len=%lld epochs=%d window=%d neg=%d lr=%.4f threads=%d\n",
            vocab_size, dim, corpus_len, epochs, window, neg_samples, lr_start, num_threads);

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        /* Thread-local state */
        rng_t rng;
        rng_seed(&rng, seed + omp_get_thread_num() * 31337ULL);

        /* Gradient buffers on stack (dim is typically ≤ 300) */
        float *grad_in  = (float *)malloc(dim * sizeof(float));
        float *grad_out = (float *)malloc(dim * sizeof(float));

        for (int ep = 0; ep < epochs; ep++) {
            /* Linear LR decay per epoch */
            float lr_f = (float)(lr_start * (1.0 - (double)ep / (double)epochs));
            if (lr_f < lr_start * 0.0001f) lr_f = lr_start * 0.0001f;
            long long ep_start = (long long)ep * corpus_len;

            #pragma omp for schedule(static)
            for (long long pos = 0; pos < corpus_len; pos++) {
                int target = corpus[pos];

                /* Subsampling */
                if (rng_uniform(&rng) > sub_prob[target])
                    continue;

                /* Dynamic window: uniform [1, window] */
                int win = rng_int_n(&rng, window) + 1;

                /* Context range */
                long long ctx_start = pos - win;
                if (ctx_start < 0) ctx_start = 0;
                long long ctx_end = pos + win + 1;
                if (ctx_end > corpus_len) ctx_end = corpus_len;

                float *v_target = w_in + (long long)target * dim;

                for (long long ctx = ctx_start; ctx < ctx_end; ctx++) {
                    if (ctx == pos) continue;

                    int context = corpus[ctx];
                    float *v_context = w_out + (long long)context * dim;

                    /* --- Positive sample --- */
                    float dot = 0.0f;
                    #pragma omp simd reduction(+:dot)
                    for (int d = 0; d < dim; d++)
                        dot += v_target[d] * v_context[d];

                    float g = (fast_sigmoid(dot, exp_table) - 1.0f) * lr_f;

                    /* Prepare gradient buffers */
                    #pragma omp simd
                    for (int d = 0; d < dim; d++) {
                        grad_in[d]  = g * v_context[d];
                        grad_out[d] = g * v_target[d];
                    }

                    /* --- Negative samples --- */
                    for (int k = 0; k < neg_samples; k++) {
                        int neg_word;
                        do {
                            neg_word = unigram_table[rng_int_n(&rng, UNIGRAM_TABLE_SIZE)];
                        } while (neg_word == context || neg_word == target);

                        float *v_neg = w_out + (long long)neg_word * dim;

                        float dot_neg = 0.0f;
                        #pragma omp simd reduction(+:dot_neg)
                        for (int d = 0; d < dim; d++)
                            dot_neg += v_target[d] * v_neg[d];

                        float g_neg = fast_sigmoid(dot_neg, exp_table) * lr_f;

                        /* Accumulate into grad_in, update v_neg (Hogwild) */
                        #pragma omp simd
                        for (int d = 0; d < dim; d++) {
                            grad_in[d] += g_neg * v_neg[d];
                            v_neg[d]   -= g_neg * v_target[d];
                        }
                    }

                    /* Apply gradients to target and positive context */
                    #pragma omp simd
                    for (int d = 0; d < dim; d++) {
                        v_target[d]  -= grad_in[d];
                        v_context[d] -= grad_out[d];
                    }
                }
            }

            /* Barrier between epochs (implicit at end of omp for) */
            if (omp_get_thread_num() == 0)
                fprintf(stderr, "Epoch %d/%d done.\n", ep + 1, epochs);
        }

        free(grad_in);
        free(grad_out);
    }

    free(unigram_table);
    free(sub_prob);
    fprintf(stderr, "Training complete.\n");
    return 0;
}