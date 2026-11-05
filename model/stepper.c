// The Doom machine's Qwen3.5 layers (model/pack.py), one token at a time, on CPU cores.
//
// The computation of transformers' Qwen3.5 decode step (Qwen3_5DecoderLayer with
// torch_recurrent_gated_delta_rule and causal_conv1d_update, bf16 with fp32 inside), with the
// weights that are zero left out: a product walks only the inputs that are nonzero and, for each,
// only the outputs its nonzero weights reach. A zero weight or input adds 0 to a sum, so skipping
// it changes nothing. Values are rounded to bf16 where torch's tensors are bf16. A value head
// whose key is zero is not written (the update adds k * delta = 0), one whose query is zero
// outputs 0, and one whose decay is exactly 1 is not scaled. Value head v reads key head
// v / (value heads / key heads), as torch's repeat_interleave makes it.
//
// Built by model/stepper.py with the Mac's clang; driven through ctypes.

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define K 128      // head dim
#define CONV 4
#define NT 4       // threads for the output head

static inline float bf(float x) {  // round to bf16, nearest even (torch's .to(bfloat16))
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return x;
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

typedef struct {  // a weight matrix by input: input j's outputs rows[colptr[j]..colptr[j+1])
    int nin, nout;
    const int32_t *colptr, *rows;
    const float *vals;
    int nlive;
    int32_t *live;  // the outputs any weight reaches, ascending
} Mat;

enum { QKV, Z, B, A, O, GATE, UP, DOWN, NMAT };

typedef struct {
    Mat m[NMAT];
    const float *conv, *A_log, *dt_bias, *gnorm, *in_norm, *post_norm;
    float *S;   // (NV, K, K) recurrent state, fp32: S[v][key][value]
    float *cs;  // (QKV channels, CONV - 1) conv state, bf16 values
    float *qkv, *z, *bb, *aa, *cv, *y, *o, *g, *u, *mm, *dn;  // stay 0 outside live outputs
    char *act;  // value heads whose output was nonzero last token
    int decays; // some value head can decay
} Layer;

typedef struct {
    int nl, D, F, NK, NV, vocab;
    Layer *L;
    const float *norm;
    const int32_t *erow, *ecol;  // embedding, CSR by token
    const float *eval;
    int ncode;
    int32_t *code;     // the code dims (the only dims the final norm keeps)
    int32_t *hptr, *hrow;  // output head: per code dim, the tokens with an entry there
    float *hval;
    float *acc, *logits, *h, *x;
    uint64_t *hz;      // the residual's nonzero dims
    int32_t *xi, *hi;
    // output head threads
    float hx[64];
    int hcount;
    _Atomic int gen, done;
    float bv[NT];
    int best[NT];
    pthread_t th[NT];
} Model;

static void mat_init(Mat *m, int nin, int nout, const int32_t *cp, const int32_t *rows,
                     const float *vals) {
    m->nin = nin, m->nout = nout, m->colptr = cp, m->rows = rows, m->vals = vals;
    char *seen = calloc(nout, 1);
    for (int i = 0; i < cp[nin]; i++) seen[rows[i]] = 1;
    m->nlive = 0;
    for (int r = 0; r < nout; r++) m->nlive += seen[r];
    m->live = malloc(sizeof(int32_t) * (m->nlive + 1));
    for (int r = 0, n = 0; r < nout; r++)
        if (seen[r]) m->live[n++] = r;
    free(seen);
}

// y = bf16(W x) on W's live outputs; x has nx nonzero entries at xi
static void matvec(Model *M, const Mat *m, const float *x, const int32_t *xi, int nx, float *y) {
    float *acc = M->acc;
    for (int r = 0; r < m->nlive; r++) acc[m->live[r]] = 0.0f;
    for (int t = 0; t < nx; t++) {
        int j = xi[t];
        float xv = x[j];
        for (int p = m->colptr[j]; p < m->colptr[j + 1]; p++) acc[m->rows[p]] += m->vals[p] * xv;
    }
    for (int r = 0; r < m->nlive; r++) y[m->live[r]] = bf(acc[m->live[r]]);
}

static inline void hz_set(Model *M, int i, float v) {
    if (v != 0.0f) M->hz[i >> 6] |= 1ull << (i & 63); else M->hz[i >> 6] &= ~(1ull << (i & 63));
}

// Qwen3_5RMSNorm: fp32 mean of squares, x * rsqrt(var + eps) * (1 + w), to bf16; walks only the
// residual's nonzero dims (the rest add +0), lists x's nonzero entries in xi
static int rmsnorm(Model *M, const float *w, float *x, int32_t *xi) {
    int n = 0, nw = (M->D + 63) / 64;
    for (int k = 0; k < nw; k++)
        for (uint64_t b = M->hz[k]; b; b &= b - 1) M->hi[n++] = k * 64 + __builtin_ctzll(b);
    float ss = 0.0f;
    for (int t = 0; t < n; t++) ss += M->h[M->hi[t]] * M->h[M->hi[t]];
    float r = 1.0f / sqrtf(ss / M->D + 1e-6f);
    int nx = 0;
    for (int t = 0; t < n; t++) {
        int i = M->hi[t];
        x[i] = bf((M->h[i] * r) * (1.0f + w[i]));
        if (x[i] != 0.0f) xi[nx++] = i;
    }
    return nx;
}

Model *st_new(int nl, int D, int F, int NK, int NV, int vocab) {
    Model *M = calloc(1, sizeof(Model));
    M->nl = nl, M->D = D, M->F = F, M->NK = NK, M->NV = NV, M->vocab = vocab;
    M->L = calloc(nl, sizeof(Layer));
    int big = 2 * NK * K + NV * K;
    if (big < F) big = F;
    if (big < D) big = D;
    M->acc = calloc(big, sizeof(float));
    M->logits = calloc(vocab, sizeof(float));
    M->h = calloc(D, sizeof(float));
    M->x = calloc(D, sizeof(float));
    M->hz = calloc((D + 63) / 64, sizeof(uint64_t));
    M->xi = malloc(sizeof(int32_t) * big);
    M->hi = malloc(sizeof(int32_t) * D);
    return M;
}

void st_mat(Model *M, int layer, int which, int nin, int nout, const int32_t *cp,
            const int32_t *rows, const float *vals) {
    mat_init(&M->L[layer].m[which], nin, nout, cp, rows, vals);
}

void st_layer(Model *M, int layer, const float *conv, const float *A_log, const float *dt_bias,
              const float *gnorm, const float *in_norm, const float *post_norm) {
    Layer *L = &M->L[layer];
    int NK = M->NK, NV = M->NV, C = 2 * NK * K + NV * K;
    L->conv = conv, L->A_log = A_log, L->dt_bias = dt_bias, L->gnorm = gnorm;
    L->in_norm = in_norm, L->post_norm = post_norm;
    L->S = calloc((size_t)NV * K * K, sizeof(float));
    L->cs = calloc((size_t)C * (CONV - 1), sizeof(float));
    L->qkv = calloc(C, sizeof(float)), L->cv = calloc(C, sizeof(float));
    L->z = calloc(NV * K, sizeof(float)), L->y = calloc(NV * K, sizeof(float));
    L->bb = calloc(NV, sizeof(float)), L->aa = calloc(NV, sizeof(float));
    L->o = calloc(M->D, sizeof(float)), L->dn = calloc(M->D, sizeof(float));
    L->g = calloc(M->F, sizeof(float)), L->u = calloc(M->F, sizeof(float));
    L->mm = calloc(M->F, sizeof(float));
    L->act = calloc(NV, 1);
    // a head decays only if its a-projection can be nonzero or its bias gives a decay below 1
    L->decays = L->m[A].nlive > 0;
    for (int v = 0; v < NV; v++)
        if (expf(-expf(A_log[v]) * softplus(dt_bias[v])) != 1.0f) L->decays = 1;
}

// the initial state of one value head (the machine's snapshot): rows[n] (K values each) at keys
void st_state_rows(Model *M, int layer, int vh, int n, const int32_t *keys, const float *rows) {
    float *S = M->L[layer].S + (size_t)vh * K * K;
    for (int i = 0; i < n; i++) memcpy(S + (size_t)keys[i] * K, rows + (size_t)i * K, K * sizeof(float));
}

float *st_state(Model *M, int layer) { return M->L[layer].S; }

void st_reset(Model *M) {
    int C = 2 * M->NK * K + M->NV * K;
    for (int l = 0; l < M->nl; l++) {
        memset(M->L[l].S, 0, sizeof(float) * M->NV * K * K);
        memset(M->L[l].cs, 0, sizeof(float) * C * (CONV - 1));
    }
}

// --- the output head: logits over the vocabulary from the code dims, in NT token ranges -------

static void head_range(Model *M, int t) {
    int v0 = (int)((long)t * M->vocab / NT), v1 = (int)((long)(t + 1) * M->vocab / NT);
    float *lg = M->logits;
    memset(lg + v0, 0, sizeof(float) * (v1 - v0));
    for (int c = 0; c < M->ncode; c++) {  // the code dims in ascending order, as the sum's terms
        float xv = M->hx[c];
        if (xv == 0.0f) continue;
        // this column's entries are sorted by token: find range [v0, v1)
        int lo = M->hptr[c], hi = M->hptr[c + 1];
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (M->hrow[mid] < v0) lo = mid + 1; else hi = mid;
        }
        for (int p = lo; p < M->hptr[c + 1] && M->hrow[p] < v1; p++) lg[M->hrow[p]] += M->hval[p] * xv;
    }
    int best = v0;
    float bv = bf(lg[v0]);
    for (int v = v0 + 1; v < v1; v++) {
        float b = bf(lg[v]);
        if (b > bv) bv = b, best = v;
    }
    M->bv[t] = bv, M->best[t] = best;
}

typedef struct { Model *M; int t; } Arg;

static void *worker(void *p) {
    Arg a = *(Arg *)p;
    free(p);
    int seen = 0;
    for (;;) {
        int g;
        while ((g = atomic_load_explicit(&a.M->gen, memory_order_acquire)) == seen)
            __asm__ volatile("yield");
        seen = g;
        head_range(a.M, a.t);
        atomic_fetch_add_explicit(&a.M->done, 1, memory_order_release);
    }
}

// the embedding (CSR) and the final norm; the output head is the embedding's columns at the code
// dims (the final norm's gain is 0 everywhere else)
void st_embed(Model *M, const int32_t *erow, const int32_t *ecol, const float *eval,
              const float *norm, int ncode, const int32_t *code) {
    M->erow = erow, M->ecol = ecol, M->eval = eval, M->norm = norm;
    M->ncode = ncode;
    M->code = malloc(sizeof(int32_t) * ncode);
    memcpy(M->code, code, sizeof(int32_t) * ncode);
    int *slot = malloc(sizeof(int) * M->D);
    for (int i = 0; i < M->D; i++) slot[i] = -1;
    for (int c = 0; c < ncode; c++) slot[code[c]] = c;
    M->hptr = calloc(ncode + 1, sizeof(int32_t));
    for (int t = 0; t < M->vocab; t++)
        for (int p = erow[t]; p < erow[t + 1]; p++)
            if (slot[ecol[p]] >= 0) M->hptr[slot[ecol[p]] + 1]++;
    for (int c = 0; c < ncode; c++) M->hptr[c + 1] += M->hptr[c];
    int *fill = calloc(ncode, sizeof(int));
    M->hrow = malloc(sizeof(int32_t) * (M->hptr[ncode] + 1));
    M->hval = malloc(sizeof(float) * (M->hptr[ncode] + 1));
    for (int t = 0; t < M->vocab; t++)  // tokens ascending, so each column is sorted
        for (int p = erow[t]; p < erow[t + 1]; p++) {
            int c = slot[ecol[p]];
            if (c < 0) continue;
            int q = M->hptr[c] + fill[c]++;
            M->hrow[q] = t, M->hval[q] = eval[p];
        }
    free(fill), free(slot);
    for (int t = 1; t < NT; t++) {
        Arg *a = malloc(sizeof(Arg));
        a->M = M, a->t = t;
        pthread_create(&M->th[t], NULL, worker, a);
    }
}

// One token through every layer; returns the argmax of the bf16 logits (the first, on ties), or
// -1 when head is 0 (a token whose prediction is not read).
int st_step(Model *M, int tok, int head) {
    float *h = M->h, *x = M->x;
    int32_t *xi = M->xi;
    int NK = M->NK, NV = M->NV, r = NV / NK, QK = NK * K;
    memset(h, 0, sizeof(float) * M->D), memset(M->hz, 0, sizeof(uint64_t) * ((M->D + 63) / 64));
    for (int p = M->erow[tok]; p < M->erow[tok + 1]; p++)
        h[M->ecol[p]] = M->eval[p], hz_set(M, M->ecol[p], M->eval[p]);

    for (int l = 0; l < M->nl; l++) {
        Layer *L = &M->L[l];
        float *qkv = L->qkv, *z = L->z, *bb = L->bb, *aa = L->aa, *cv = L->cv, *y = L->y;
        // --- token mixer: Gated DeltaNet
        int nx = rmsnorm(M, L->in_norm, x, xi);
        matvec(M, &L->m[QKV], x, xi, nx, qkv);
        matvec(M, &L->m[Z], x, xi, nx, z);
        matvec(M, &L->m[B], x, xi, nx, bb);
        matvec(M, &L->m[A], x, xi, nx, aa);
        for (int t = 0; t < L->m[QKV].nlive; t++) {  // causal_conv1d_update, silu, in bf16
            int c = L->m[QKV].live[t];
            float *s = L->cs + c * (CONV - 1);
            const float *w = L->conv + c * CONV;
            float acc = s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + qkv[c] * w[3];
            cv[c] = bf(silu(bf(acc)));
            s[0] = s[1], s[1] = s[2], s[2] = qkv[c];
        }
        // per key head: the l2-normed key and query (shared by its r value heads)
        int32_t *yi = M->hi;  // (free here: rmsnorm's list is used up)
        int ny = 0;
        for (int j = 0; j < NK; j++) {
            float *q = cv + j * K, *k = cv + QK + j * K;
            int32_t ki[K], qi[K];
            int nk = 0, nq = 0;
            float kss = 0.0f, qss = 0.0f, kf[K], qf[K];
            for (int i = 0; i < K; i++) {
                kss += k[i] * k[i], qss += q[i] * q[i];
                if (k[i] != 0.0f) ki[nk++] = i;
                if (q[i] != 0.0f) qi[nq++] = i;
            }
            float kn = 1.0f / sqrtf(kss + 1e-6f), qn = 1.0f / sqrtf(qss + 1e-6f);
            for (int t = 0; t < nk; t++) kf[ki[t]] = k[ki[t]] * kn;
            for (int t = 0; t < nq; t++) qf[qi[t]] = (q[qi[t]] * qn) / 11.313708498984761f;
            for (int x2 = 0; x2 < r; x2++) {
                int vh = j * r + x2;
                float *v = cv + 2 * QK + vh * K;
                float *S = L->S + (size_t)vh * K * K;
                if (L->decays) {
                    float gg = -expf(L->A_log[vh]) * softplus(aa[vh] + L->dt_bias[vh]);
                    float decay = expf(gg);
                    if (decay != 1.0f)
                        for (int i = 0; i < K * K; i++) S[i] *= decay;
                }
                if (nk) {
                    float beta = bf(sigm(bb[vh])), kv[K], delta[K];
                    for (int c = 0; c < K; c++) kv[c] = 0.0f;
                    for (int t = 0; t < nk; t++) {
                        const float *row = S + ki[t] * K;
                        float kk = kf[ki[t]];
                        for (int c = 0; c < K; c++) kv[c] += row[c] * kk;
                    }
                    for (int c = 0; c < K; c++) delta[c] = (v[c] - kv[c]) * beta;
                    for (int t = 0; t < nk; t++) {
                        float *row = S + ki[t] * K;
                        float kk = kf[ki[t]];
                        for (int c = 0; c < K; c++) row[c] = row[c] + kk * delta[c];
                    }
                }
                float *yy = y + vh * K;
                if (!nq) {  // a zero query reads 0, and the gated norm of 0 is 0
                    if (L->act[vh]) memset(yy, 0, sizeof(float) * K), L->act[vh] = 0;
                    continue;
                }
                L->act[vh] = 1;
                float out[K], ss = 0.0f;
                for (int c = 0; c < K; c++) out[c] = 0.0f;
                for (int t = 0; t < nq; t++) {
                    const float *row = S + qi[t] * K;
                    float qq = qf[qi[t]];
                    for (int c = 0; c < K; c++) out[c] += row[c] * qq;
                }
                for (int c = 0; c < K; c++) out[c] = bf(out[c]), ss += out[c] * out[c];
                float rn = 1.0f / sqrtf(ss / K + 1e-6f), *zz = z + vh * K;
                for (int c = 0; c < K; c++) {  // Qwen3_5RMSNormGated
                    float t2 = bf(out[c] * rn);
                    t2 = bf(L->gnorm[c] * t2);
                    yy[c] = t2 == 0.0f ? 0.0f : bf(t2 * silu(zz[c]));
                    if (yy[c] != 0.0f) yi[ny++] = vh * K + c;
                }
            }
        }
        matvec(M, &L->m[O], y, yi, ny, L->o);
        for (int t = 0; t < L->m[O].nlive; t++) {
            int i = L->m[O].live[t];
            h[i] = bf(h[i] + L->o[i]), hz_set(M, i, h[i]);
        }
        // --- MLP
        nx = rmsnorm(M, L->post_norm, x, xi);
        matvec(M, &L->m[GATE], x, xi, nx, L->g);
        matvec(M, &L->m[UP], x, xi, nx, L->u);
        int nm = 0;
        for (int t = 0; t < L->m[GATE].nlive; t++) {
            int i = L->m[GATE].live[t];
            L->mm[i] = L->u[i] == 0.0f ? 0.0f : bf(bf(silu(L->g[i])) * L->u[i]);
            if (L->mm[i] != 0.0f) xi[nm++] = i;
        }
        matvec(M, &L->m[DOWN], L->mm, xi, nm, L->dn);
        for (int t = 0; t < L->m[DOWN].nlive; t++) {
            int i = L->m[DOWN].live[t];
            h[i] = bf(h[i] + L->dn[i]), hz_set(M, i, h[i]);
        }
    }
    if (!head) return -1;
    // the final norm (its gain is 0 outside the code dims) and the output head
    rmsnorm(M, M->norm, x, xi);
    for (int c = 0; c < M->ncode; c++) M->hx[c] = (M->hz[M->code[c] >> 6] >> (M->code[c] & 63) & 1) ? x[M->code[c]] : 0.0f;
    atomic_store_explicit(&M->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&M->gen, 1, memory_order_release);
    head_range(M, 0);
    while (atomic_load_explicit(&M->done, memory_order_acquire) < NT - 1) __asm__ volatile("yield");
    int best = M->best[0];
    float bv = M->bv[0];
    for (int t = 1; t < NT; t++)
        if (M->bv[t] > bv) bv = M->bv[t], best = M->best[t];
    return best;
}

// Feed n tokens; the prediction after the last is the first generated token. Generate until a
// token in stop[0..ns), at most max_new; returns how many were generated (into out).
int st_run(Model *M, const int32_t *toks, int n, const int32_t *stop, int ns, long max_new,
           int32_t *out) {
    int t = 0;
    for (int i = 0; i < n; i++) t = st_step(M, toks[i], i == n - 1);
    long g = 0;
    while (g < max_new) {
        out[g++] = t;
        for (int s = 0; s < ns; s++)
            if (t == stop[s]) return (int)g;
        t = st_step(M, t, 1);
    }
    return (int)g;
}
