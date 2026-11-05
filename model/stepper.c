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
// A token runs on NT threads, each layer in four phases with a barrier after each: the token
// mixer by groups of key heads (each group's slices of the projections, its conv channels, its
// value heads), then the out projection, the MLP's gate and up, and its down projection, each by
// ranges of output rows. Every output is still summed by one thread over its inputs in the same
// order, so the result is the same, bit for bit, as on one thread.
//
// Built by model/stepper.py with the Mac's clang; driven through ctypes.

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 128      // head dim
#define CONV 4
#define NT 4       // threads (the Mac's performance cores)

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
    uint16_t *w16;  // the kept weights as bf16 bits, in run order (run k's at rpos[k])
    int nlive;
    int32_t *live;  // the outputs any weight reaches, ascending
    // each input's outputs as runs of consecutive rows (rows ascend within a column, so a run's
    // weights are consecutive too): run k of input j is rrow[k], rpos[k], rlen[k] for
    // k in rptr[j]..rptr[j+1]; the live outputs as runs too
    int32_t *rptr, *rrow, *rpos, *rlen;
    int nlr;
    int32_t *lrow, *llen;
    int ncol;
    int32_t *col;   // the inputs with any weight, ascending
    // the ballast inputs (the first nb dims: they come first in every sum, and never change):
    // each live output's partial sum over them, made for the ballast values bx
    int nb, base_ok;
    float bx[16], *base;
} Mat;

enum { QKV, Z, B, A, O, GATE, UP, DOWN, NMAT };

typedef struct {
    Mat m[NMAT];       // whole (only its live outputs kept, for counts)
    Mat sub[NMAT][NT]; // thread t's part: its key heads' rows of QKV/Z/B/A, its row ranges of the rest
    int nh[NT], heads[NT][64];  // thread t's key heads
    int fstart[NT + 1];         // thread t's MLP units are [fstart[t], fstart[t + 1])
    const float *conv, *A_log, *dt_bias, *gnorm, *in_norm, *post_norm;
    float *S;   // (NV, K, K) recurrent state, fp32: S[v][key][value]
    float *cs;  // conv state: the QKV outputs of the last CONV - 1 tokens, token T's in plane T % 3
    // thread t's conv channels as segments of one tap pattern and the same weights: segment k is
    // channels cc0[k] .. + cn[k], taps cmask[k] (bit i: tap i, 0 the oldest, has a nonzero weight),
    // weights cwt[k][0..3]
    int nseg[NT];
    int32_t *cc0[NT], *cn[NT];
    uint8_t *cmask[NT];
    float (*cwt[NT])[4];
    float *qkv, *z, *bb, *aa, *cv, *y, *o, *g, *u, *mm, *dn;  // stay 0 outside live outputs
    int32_t *yl, *yn;  // value head v's nonzero outputs: yl[v * K ..], yn[v] of them
    int32_t *ml, mn[NT];  // thread t's nonzero MLP units: ml[fstart[t] ..], mn[t] of them
    char *act;  // value heads whose output was nonzero last token
    int decays; // some value head can decay
    int nomlp;  // the MLP has no weights
} Layer;

typedef struct {  // a thread's own buffers
    float *acc;
    int32_t *li;
    int sense;
} Ctx;

typedef struct {
    int nl, D, F, NK, NV, vocab, nballast;
    Layer *L;
    const float *norm;
    const int32_t *erow, *ecol;  // embedding, CSR by token
    const float *eval;
    int ncode;
    int32_t *code;     // the code dims (the only dims the final norm keeps)
    int32_t *hptr, *hrow;  // output head: per code dim, the tokens with an entry there
    float *hval;
    float *logits, *h;
    uint64_t *hz;      // the residual's nonzero dims
    Ctx cx[NT];
    float hx[64];
    // the threads: a token starts when gen moves; done counts the workers finished
    int head, tick;    // tick: tokens since the reset (picks the conv state planes)
    _Atomic int gen, done, bcount, bsense, nonfinite;
    float bv[NT];
    int best[NT];
    pthread_t th[NT];
} Model;

static void bar(Model *M, Ctx *c) {  // all NT threads meet
    int s = c->sense ^= 1;
    if (atomic_fetch_add_explicit(&M->bcount, 1, memory_order_acq_rel) == NT - 1) {
        atomic_store_explicit(&M->bcount, 0, memory_order_relaxed);
        atomic_store_explicit(&M->bsense, s, memory_order_release);
    } else {
        while (atomic_load_explicit(&M->bsense, memory_order_acquire) != s) __asm__ volatile("yield");
    }
}

static void mat_init(Mat *m, int nin, int nout, const int32_t *cp, const int32_t *rows,
                     const float *vals, const char *own, int g) {
    m->nin = nin, m->nout = nout, m->colptr = cp, m->rows = rows, m->vals = vals;
    char *seen = calloc(nout, 1);
    for (int i = 0; i < cp[nin]; i++)
        if (!own || own[rows[i]] == g) seen[rows[i]] = 1;
    m->nlive = 0;
    for (int r = 0; r < nout; r++) m->nlive += seen[r];
    m->live = malloc(sizeof(int32_t) * (m->nlive + 1));
    for (int r = 0, n = 0; r < nout; r++)
        if (seen[r]) m->live[n++] = r;
    free(seen);
    int nr = 0;
    m->rptr = malloc(sizeof(int32_t) * (nin + 1));
    m->rrow = malloc(sizeof(int32_t) * (cp[nin] + 1));
    m->rpos = malloc(sizeof(int32_t) * (cp[nin] + 1));
    m->rlen = malloc(sizeof(int32_t) * (cp[nin] + 1));
    int nkept = 0;
    for (int j = 0; j < nin; j++) {
        m->rptr[j] = nr;
        for (int p = cp[j], prev = -2; p < cp[j + 1]; p++) {
            if (own && own[rows[p]] != g) continue;
            if (prev == p - 1 && rows[p] == rows[p - 1] + 1) m->rlen[nr - 1]++;
            else m->rrow[nr] = rows[p], m->rpos[nr] = p, m->rlen[nr] = 1, nr++;
            prev = p, nkept++;
        }
    }
    m->rptr[nin] = nr;
    m->w16 = malloc(sizeof(uint16_t) * (nkept + 1));
    for (int k = 0, o = 0; k < nr; k++) {  // every weight is a bf16 value (model/stepper.py checks)
        for (int i = 0; i < m->rlen[k]; i++) {
            uint32_t u;
            memcpy(&u, &vals[m->rpos[k] + i], 4);
            m->w16[o + i] = (uint16_t)(u >> 16);
        }
        m->rpos[k] = o, o += m->rlen[k];
    }
    m->col = malloc(sizeof(int32_t) * (nin + 1)), m->ncol = 0;
    for (int j = 0; j < nin; j++)
        if (m->rptr[j + 1] > m->rptr[j]) m->col[m->ncol++] = j;
    m->rrow = realloc(m->rrow, sizeof(int32_t) * (nr + 1));
    m->rpos = realloc(m->rpos, sizeof(int32_t) * (nr + 1));
    m->rlen = realloc(m->rlen, sizeof(int32_t) * (nr + 1));
    m->lrow = malloc(sizeof(int32_t) * (m->nlive + 1));
    m->llen = malloc(sizeof(int32_t) * (m->nlive + 1));
    m->nlr = 0;
    for (int r = 0; r < m->nlive; r++) {
        if (r && m->live[r] == m->live[r - 1] + 1) m->llen[m->nlr - 1]++;
        else m->lrow[m->nlr] = m->live[r], m->llen[m->nlr] = 1, m->nlr++;
    }
}

static void mat_free_runs(Mat *m) {
    free(m->rptr), free(m->rrow), free(m->rpos), free(m->rlen), free(m->lrow), free(m->llen);
    free(m->col), free(m->w16);
    m->w16 = NULL;
    m->rptr = m->rrow = m->rpos = m->rlen = m->lrow = m->llen = m->col = NULL;
}

// acc += W[:, j] * xv, run by run: each output still gets its terms in input order
static inline void addcol(const Mat *m, int j, float xv, float *acc) {
    for (int k = m->rptr[j]; k < m->rptr[j + 1]; k++) {
        float *a = acc + m->rrow[k];
        const uint16_t *v = m->w16 + m->rpos[k];
        for (int i = 0, n = m->rlen[k]; i < n; i++) {
            uint32_t u = (uint32_t)v[i] << 16;
            float w;
            memcpy(&w, &u, 4);
            a[i] += w * xv;
        }
    }
}
static inline void live_fill(const Mat *m, float *acc, const float *from) {
    for (int k = 0, o = 0; k < m->nlr; k++) {
        float *a = acc + m->lrow[k];
        int n = m->llen[k];
        if (from) memcpy(a, from + o, sizeof(float) * n); else memset(a, 0, sizeof(float) * n);
        o += n;
    }
}

// y = bf16(W x) on W's live outputs; x has nx nonzero entries at xi (ascending)
static void matvec(const Mat *m, const float *x, const int32_t *xi, int nx, float *y, float *acc) {
    live_fill(m, acc, NULL);
    for (int t = 0; t < nx; t++) addcol(m, xi[t], x[xi[t]], acc);
    for (int k = 0; k < m->nlr; k++)
        for (int i = m->lrow[k], e = i + m->llen[k]; i < e; i++) y[i] = bf(acc[i]);
}

static inline float normed(const Model *M, const float *w, float r, int j);

// y = bf16(W rmsnorm(h)) on W's live outputs, reading only the inputs W has weights for: each
// normed value made where it is used (r is the norm's scale). The sums over the ballast inputs,
// which lead every sum, are kept from the last token with the same values.
static void matvec_n(const Model *M, const Mat *m_, const float *w, float r, float *y, float *acc) {
    Mat *m = (Mat *)m_;
    int c0 = 0;
    if (m->nb) {
        int nc = 0, same = m->base_ok;
        float xb[16];
        while (nc < m->ncol && m->col[nc] < m->nb) nc++;
        for (int c = 0; c < nc; c++) xb[c] = normed(M, w, r, m->col[c]), same &= xb[c] == m->bx[c];
        if (same) {
            live_fill(m, acc, m->base);
        } else {  // make the partial sums over the ballast inputs
            live_fill(m, acc, NULL);
            for (int c = 0; c < nc; c++) {
                m->bx[c] = xb[c];
                if (xb[c] != 0.0f) addcol(m, m->col[c], xb[c], acc);
            }
            if (!m->base) m->base = malloc(sizeof(float) * (m->nlive + 1));
            for (int k = 0, o = 0; k < m->nlr; k++, o += m->llen[k - 1])
                memcpy(m->base + o, acc + m->lrow[k], sizeof(float) * m->llen[k]);
            m->base_ok = 1;
        }
        c0 = nc;
    } else {
        live_fill(m, acc, NULL);
    }
    for (int c = c0; c < m->ncol; c++) {
        int j = m->col[c];
        float xv = normed(M, w, r, j);
        if (xv != 0.0f) addcol(m, j, xv, acc);
    }
    for (int k = 0; k < m->nlr; k++)
        for (int i = m->lrow[k], e = i + m->llen[k]; i < e; i++) y[i] = bf(acc[i]);
}

static inline void hz_set(Model *M, int i, float v) {
    if (v != 0.0f) M->hz[i >> 6] |= 1ull << (i & 63); else M->hz[i >> 6] &= ~(1ull << (i & 63));
}

// Qwen3_5RMSNorm: fp32 mean of squares, x * rsqrt(var + eps) * (1 + w), to bf16. The scale
// walks only the residual's nonzero dims (the rest add +0); normed() is one dim's value.
static float rms_scale(const Model *M) {
    float ss = 0.0f;
    for (int k = 0, nw = (M->D + 63) / 64; k < nw; k++)
        for (uint64_t b = M->hz[k]; b; b &= b - 1) {
            float v = M->h[k * 64 + __builtin_ctzll(b)];
            ss += v * v;
        }
    return 1.0f / sqrtf(ss / M->D + 1e-6f);
}
static inline float normed(const Model *M, const float *w, float r, int j) {
    return (M->hz[j >> 6] >> (j & 63) & 1) ? bf((M->h[j] * r) * (1.0f + w[j])) : 0.0f;
}

Model *st_new(int nl, int D, int F, int NK, int NV, int vocab) {
    Model *M = calloc(1, sizeof(Model));
    M->nl = nl, M->D = D, M->F = F, M->NK = NK, M->NV = NV, M->vocab = vocab;
    M->L = calloc(nl, sizeof(Layer));
    int big = 2 * NK * K + NV * K;
    if (big < F) big = F;
    if (big < D) big = D;
    M->logits = calloc(vocab, sizeof(float));
    M->h = calloc(D, sizeof(float));
    M->hz = calloc((D + 63) / 64, sizeof(uint64_t));
    for (int t = 0; t < NT; t++) {
        Ctx *c = &M->cx[t];
        c->acc = calloc(big, sizeof(float)), c->li = malloc(sizeof(int32_t) * big);
    }
    return M;
}

void st_mat(Model *M, int layer, int which, int nin, int nout, const int32_t *cp,
            const int32_t *rows, const float *vals) {
    mat_init(&M->L[layer].m[which], nin, nout, cp, rows, vals, NULL, 0);
    M->L[layer].m[which].nb = M->nballast;
}

// the ballast dims are 0..n-1 (model/pack.py's layout: they lead the residual)
void st_ballast(Model *M, int n) { M->nballast = n < 16 ? n : 0; }

// own[i] = the thread of row i, in NT contiguous ranges of about equal weight, cut at multiples
// of align
static void ranges(int n, const double *w, int align, char *own) {
    double tot = 0, cum = 0;
    for (int i = 0; i < n; i++) tot += w[i];
    for (int i0 = 0; i0 < n; i0 += align) {
        int e = i0 + align < n ? i0 + align : n, t = tot > 0 ? (int)(cum * NT / tot) : 0;
        if (t >= NT) t = NT - 1;
        for (int i = i0; i < e; i++) own[i] = (char)t, cum += w[i];
    }
}

static void row_weight(const Mat *m, double *w) {
    for (int p = 0; p < m->colptr[m->nin]; p++) w[m->rows[p]] += 1;
}

void st_layer(Model *M, int layer, const float *conv, const float *A_log, const float *dt_bias,
              const float *gnorm, const float *in_norm, const float *post_norm) {
    Layer *L = &M->L[layer];
    int NK = M->NK, NV = M->NV, r = NV / NK, QK = NK * K, C = 2 * QK + NV * K, D = M->D, F = M->F;
    L->conv = conv, L->A_log = A_log, L->dt_bias = dt_bias, L->gnorm = gnorm;
    L->in_norm = in_norm, L->post_norm = post_norm;
    L->S = calloc((size_t)NV * K * K, sizeof(float));
    L->cs = calloc((size_t)C * (CONV - 1), sizeof(float));
    L->qkv = calloc(C, sizeof(float)), L->cv = calloc(C, sizeof(float));
    L->z = calloc(NV * K, sizeof(float)), L->y = calloc(NV * K, sizeof(float));
    L->bb = calloc(NV, sizeof(float)), L->aa = calloc(NV, sizeof(float));
    L->o = calloc(D, sizeof(float)), L->dn = calloc(D, sizeof(float));
    L->g = calloc(F, sizeof(float)), L->u = calloc(F, sizeof(float));
    L->mm = calloc(F, sizeof(float));
    L->yl = malloc(sizeof(int32_t) * NV * K), L->yn = calloc(NV, sizeof(int32_t));
    L->ml = malloc(sizeof(int32_t) * F);
    L->act = calloc(NV, 1);
    // a head decays only if its a-projection can be nonzero or its bias gives a decay below 1
    L->decays = L->m[A].nlive > 0;
    for (int v = 0; v < NV; v++)
        if (expf(-expf(A_log[v]) * softplus(dt_bias[v])) != 1.0f) L->decays = 1;

    // the key heads into NT groups of about equal work (largest first, to the lightest group)
    double cost[64] = {0}, load[NT] = {0};
    {
        double *w = calloc(C, sizeof(double)), *wz = calloc(NV * K, sizeof(double));
        double *wb = calloc(NV, sizeof(double));
        row_weight(&L->m[QKV], w), row_weight(&L->m[Z], wz);
        row_weight(&L->m[B], wb), row_weight(&L->m[A], wb);
        char *live = calloc(C, 1);
        for (int t = 0; t < L->m[QKV].nlive; t++) live[L->m[QKV].live[t]] = 1;
        for (int c = 0; c < C; c++) {
            int j = c < QK ? c / K : c < 2 * QK ? (c - QK) / K : (c - 2 * QK) / K / r;
            cost[j] += w[c] + 8 * live[c];
            if (c >= QK && c < 2 * QK && live[c]) cost[j] += 2.0 * r * K;  // the delta update
        }
        for (int o = 0; o < NV * K; o++) cost[o / K / r] += wz[o];
        for (int v = 0; v < NV; v++) cost[v / r] += wb[v] + (L->decays ? K * K / 4.0 : 0);
        free(w), free(wz), free(wb), free(live);
    }
    char hg[64];
    {
        char done[64] = {0};
        for (int n = 0; n < NK; n++) {
            int j = -1;
            for (int i = 0; i < NK; i++)
                if (!done[i] && (j < 0 || cost[i] > cost[j])) j = i;
            int t = 0;
            for (int u = 1; u < NT; u++)
                if (load[u] < load[t]) t = u;
            done[j] = 1, hg[j] = (char)t, load[t] += cost[j];
        }
        for (int t = 0; t < NT; t++) L->nh[t] = 0;
        for (int j = 0; j < NK; j++) L->heads[(int)hg[j]][L->nh[(int)hg[j]]++] = j;
    }
    char *own = malloc(C > F ? (C > D ? C : D) : (F > D ? F : D));
    double *w = malloc(sizeof(double) * (C > F ? (C > D ? C : D) : (F > D ? F : D)));
    for (int t = 0; t < NT; t++) {
        const Mat *f;
        for (int c = 0; c < C; c++)
            own[c] = hg[c < QK ? c / K : c < 2 * QK ? (c - QK) / K : (c - 2 * QK) / K / r];
        f = &L->m[QKV], mat_init(&L->sub[QKV][t], f->nin, f->nout, f->colptr, f->rows, f->vals, own, t);
        for (int o = 0; o < NV * K; o++) own[o] = hg[o / K / r];
        f = &L->m[Z], mat_init(&L->sub[Z][t], f->nin, f->nout, f->colptr, f->rows, f->vals, own, t);
        for (int v = 0; v < NV; v++) own[v] = hg[v / r];
        f = &L->m[B], mat_init(&L->sub[B][t], f->nin, f->nout, f->colptr, f->rows, f->vals, own, t);
        f = &L->m[A], mat_init(&L->sub[A][t], f->nin, f->nout, f->colptr, f->rows, f->vals, own, t);
    }
    // the out and down projections by 64-row ranges of the residual (a thread then owns whole
    // words of its nonzero bitmap), gate and up by ranges of units
    int which[3] = {O, DOWN, GATE};
    for (int k = 0; k < 3; k++) {
        int m = which[k], n = L->m[m].nout;
        memset(w, 0, sizeof(double) * n);
        row_weight(&L->m[m], w);
        if (m == GATE) row_weight(&L->m[UP], w);
        ranges(n, w, m == GATE ? 1 : 64, own);
        for (int t = 0; t < NT; t++) {
            const Mat *f = &L->m[m];
            mat_init(&L->sub[m][t], f->nin, f->nout, f->colptr, f->rows, f->vals, own, t);
            if (m == GATE)
                f = &L->m[UP], mat_init(&L->sub[UP][t], f->nin, f->nout, f->colptr, f->rows, f->vals, own, t);
        }
        if (m == GATE) {
            for (int t = 0; t <= NT; t++) L->fstart[t] = n;
            for (int i = n - 1; i >= 0; i--) L->fstart[(int)own[i]] = i;
            for (int t = NT - 1; t >= 0; t--)
                if (L->fstart[t] > L->fstart[t + 1]) L->fstart[t] = L->fstart[t + 1];
        }
    }
    for (int t = 0; t < NT; t++) {  // the conv segments, over thread t's live QKV channels
        const Mat *mq = &L->sub[QKV][t];
        int n = 0;
        L->cc0[t] = malloc(sizeof(int32_t) * (mq->nlive + 1)), L->cn[t] = malloc(sizeof(int32_t) * (mq->nlive + 1));
        L->cmask[t] = malloc(mq->nlive + 1), L->cwt[t] = malloc(sizeof(float[4]) * (mq->nlive + 1));
        for (int u = 0; u < mq->nlive; u++) {
            int c = mq->live[u], mask = 0;
            const float *w = conv + c * CONV;
            for (int k = 0; k < CONV; k++) mask |= (w[k] != 0.0f) << k;
            if (n && L->cc0[t][n - 1] + L->cn[t][n - 1] == c && L->cmask[t][n - 1] == mask &&
                !memcmp(L->cwt[t][n - 1], w, sizeof(float) * CONV)) {
                L->cn[t][n - 1]++;
                continue;
            }
            L->cc0[t][n] = c, L->cn[t][n] = 1, L->cmask[t][n] = (uint8_t)mask;
            memcpy(L->cwt[t][n], w, sizeof(float) * CONV);
            n++;
        }
        L->nseg[t] = n;
    }
    L->nomlp = L->m[GATE].nlive == 0 && L->m[UP].nlive == 0 && L->m[DOWN].nlive == 0;
    for (int m = 0; m < NMAT; m++) {
        mat_free_runs(&L->m[m]);
        for (int t = 0; t < NT; t++) L->sub[m][t].nb = L->m[m].nb;
    }
    free(own), free(w);
}

// the initial state of one value head (the machine's snapshot): rows[n] (K values each) at keys
void st_state_rows(Model *M, int layer, int vh, int n, const int32_t *keys, const float *rows) {
    float *S = M->L[layer].S + (size_t)vh * K * K;
    for (int i = 0; i < n; i++) memcpy(S + (size_t)keys[i] * K, rows + (size_t)i * K, K * sizeof(float));
}

float *st_state(Model *M, int layer) { return M->L[layer].S; }

int st_nonfinite(Model *M) { return M->nonfinite; }

// The state a run carries from token to token (the recurrent states, the conv states and which
// value heads wrote an output, and the token count), to a file and back; 0 on success.
int st_save(Model *M, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int C = 2 * M->NK * K + M->NV * K, hdr[6] = {0x51444d31, M->nl, M->NV, K, C, M->tick};
    int ok = fwrite(hdr, sizeof hdr, 1, f) == 1;
    for (int l = 0; l < M->nl && ok; l++) {
        Layer *L = &M->L[l];
        ok = fwrite(L->S, sizeof(float), (size_t)M->NV * K * K, f) == (size_t)M->NV * K * K &&
             fwrite(L->cs, sizeof(float), (size_t)C * (CONV - 1), f) == (size_t)C * (CONV - 1) &&
             fwrite(L->act, 1, M->NV, f) == (size_t)M->NV;
    }
    return fclose(f) == 0 && ok ? 0 : -1;
}

int st_load(Model *M, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int C = 2 * M->NK * K + M->NV * K, hdr[6];
    int ok = fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x51444d31 && hdr[1] == M->nl &&
             hdr[2] == M->NV && hdr[3] == K && hdr[4] == C;
    for (int l = 0; l < M->nl && ok; l++) {
        Layer *L = &M->L[l];
        ok = fread(L->S, sizeof(float), (size_t)M->NV * K * K, f) == (size_t)M->NV * K * K &&
             fread(L->cs, sizeof(float), (size_t)C * (CONV - 1), f) == (size_t)C * (CONV - 1) &&
             fread(L->act, 1, M->NV, f) == (size_t)M->NV;
        memset(L->y, 0, sizeof(float) * M->NV * K);  // (a head marked active rewrites or clears it)
    }
    fclose(f);
    if (ok) M->tick = hdr[5];
    return ok ? 0 : -1;
}

void st_reset(Model *M) {
    int C = 2 * M->NK * K + M->NV * K;
    M->tick = 0;
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

// Thread t's share of every layer. Each phase reads what the phase before wrote (through the
// barrier) and writes only rows it owns.
static void layers(Model *M, int t) {
    Ctx *c = &M->cx[t];
    float *h = M->h, *acc = c->acc;
    int32_t *li = c->li;
    int NK = M->NK, NV = M->NV, r = NV / NK, QK = NK * K;
    for (int l = 0; l < M->nl; l++) {
        Layer *L = &M->L[l];
        float *qkv = L->qkv, *z = L->z, *bb = L->bb, *aa = L->aa, *cv = L->cv, *y = L->y;
        // --- token mixer: Gated DeltaNet, for this thread's key heads
        float rs = rms_scale(M);
        matvec_n(M, &L->sub[QKV][t], L->in_norm, rs, qkv, acc);
        matvec_n(M, &L->sub[Z][t], L->in_norm, rs, z, acc);
        matvec_n(M, &L->sub[B][t], L->in_norm, rs, bb, acc);
        matvec_n(M, &L->sub[A][t], L->in_norm, rs, aa, acc);
        {  // causal_conv1d_update and silu, in bf16, segment by segment. The sum is torch's,
           // ((s0 w0 + s1 w1) + s2 w2) + x w3, with the zero-weight terms left out: each is a
           // zero (the states are finite; st_nonfinite counts any that is not), so leaving it out
           // changes at most the sign of a zero sum. Only channels with a past tap keep a state.
            int C = 2 * NK * K + NV * K, T = M->tick;
            float *p3 = L->cs + (size_t)(T % 3) * C;        // 3 tokens ago, becomes this token's
            float *p2 = L->cs + (size_t)((T + 1) % 3) * C;  // 2 ago
            float *p1 = L->cs + (size_t)((T + 2) % 3) * C;  // 1 ago
            float tmp[4096];
            for (int k = 0; k < L->nseg[t]; k++) {
                const float *w = L->cwt[t][k];
                int mask = L->cmask[t][k];
                for (int c0 = L->cc0[t][k], e = c0 + L->cn[t][k]; c0 < e; c0 += 4096) {
                    int n = e - c0 < 4096 ? e - c0 : 4096;
                    const float *q = qkv + c0;
                    float *s3 = p3 + c0, *s2 = p2 + c0, *s1 = p1 + c0;
                    switch (mask) {
                    case 8:
                        for (int i = 0; i < n; i++) tmp[i] = bf(q[i] * w[3]);
                        break;
                    case 4:
                        for (int i = 0; i < n; i++) tmp[i] = bf(s1[i] * w[2]), s3[i] = q[i];
                        break;
                    case 9:
                        for (int i = 0; i < n; i++) tmp[i] = bf(s3[i] * w[0] + q[i] * w[3]), s3[i] = q[i];
                        break;
                    case 10:
                        for (int i = 0; i < n; i++) tmp[i] = bf(s2[i] * w[1] + q[i] * w[3]), s3[i] = q[i];
                        break;
                    default:
                        for (int i = 0; i < n; i++)
                            tmp[i] = bf(s3[i] * w[0] + s2[i] * w[1] + s1[i] * w[2] + q[i] * w[3]), s3[i] = q[i];
                    }
                    uint32_t top = 0;  // the largest |x| bits: inf or NaN is >= 0x7f800000
                    for (int i = 0; i < n; i++) {
                        uint32_t u;
                        memcpy(&u, &q[i], 4);
                        u &= 0x7fffffff;
                        top = u > top ? u : top;
                    }
                    if (top >= 0x7f800000) atomic_fetch_add_explicit(&M->nonfinite, 1, memory_order_relaxed);
                    // below -89, expf(-x) overflows to inf and x / inf is -0.0 (the same bits the
                    // full expression gives): nearly every unmatched address row skips expf. A
                    // zero gives itself (+-0 / 2). The rest go through silu, and they take few
                    // values: the last one's result is kept
                    float *o = cv + c0;
                    int rest = 0;
                    for (int i = 0; i < n; i++) {
                        float x = tmp[i];
                        o[i] = x < -89.0f ? -0.0f : x;
                        rest |= x >= -89.0f && x != 0.0f;
                    }
                    if (rest) {
                        float lx = 0.0f, ly = 0.0f;
                        for (int i = 0; i < n; i++) {
                            float x = tmp[i];
                            if (x < -89.0f || x == 0.0f) continue;
                            if (x != lx) lx = x, ly = bf(silu(x));
                            o[i] = ly;
                        }
                    }
                }
            }
        }
        // per key head: the l2-normed key and query (shared by its r value heads)
        for (int hh = 0; hh < L->nh[t]; hh++) {
            int j = L->heads[t][hh];
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
            for (int u = 0; u < nk; u++) kf[ki[u]] = k[ki[u]] * kn;
            for (int u = 0; u < nq; u++) qf[qi[u]] = (q[qi[u]] * qn) / 11.313708498984761f;
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
                    for (int cc = 0; cc < K; cc++) kv[cc] = 0.0f;
                    for (int u = 0; u < nk; u++) {
                        const float *row = S + ki[u] * K;
                        float kk = kf[ki[u]];
                        for (int cc = 0; cc < K; cc++) kv[cc] += row[cc] * kk;
                    }
                    for (int cc = 0; cc < K; cc++) delta[cc] = (v[cc] - kv[cc]) * beta;
                    for (int u = 0; u < nk; u++) {
                        float *row = S + ki[u] * K;
                        float kk = kf[ki[u]];
                        for (int cc = 0; cc < K; cc++) row[cc] = row[cc] + kk * delta[cc];
                    }
                }
                float *yy = y + vh * K;
                int32_t *yl = L->yl + vh * K;
                L->yn[vh] = 0;
                if (!nq) {  // a zero query reads 0, and the gated norm of 0 is 0
                    if (L->act[vh]) memset(yy, 0, sizeof(float) * K), L->act[vh] = 0;
                    continue;
                }
                L->act[vh] = 1;
                float out[K], ss = 0.0f;
                for (int cc = 0; cc < K; cc++) out[cc] = 0.0f;
                for (int u = 0; u < nq; u++) {
                    const float *row = S + qi[u] * K;
                    float qq = qf[qi[u]];
                    for (int cc = 0; cc < K; cc++) out[cc] += row[cc] * qq;
                }
                for (int cc = 0; cc < K; cc++) out[cc] = bf(out[cc]), ss += out[cc] * out[cc];
                float rn = 1.0f / sqrtf(ss / K + 1e-6f), *zz = z + vh * K;
                int ny = 0;
                for (int cc = 0; cc < K; cc++) {  // Qwen3_5RMSNormGated
                    float t2 = bf(out[cc] * rn);
                    t2 = bf(L->gnorm[cc] * t2);
                    yy[cc] = t2 == 0.0f ? 0.0f : bf(t2 * silu(zz[cc]));
                    if (yy[cc] != 0.0f) yl[ny++] = vh * K + cc;
                }
                L->yn[vh] = ny;
            }
        }
        bar(M, c);
        // --- out projection, this thread's residual rows
        int ny = 0;
        for (int vh = 0; vh < NV; vh++)
            for (int u = 0; u < L->yn[vh]; u++) li[ny++] = L->yl[vh * K + u];
        const Mat *mo = &L->sub[O][t];
        matvec(mo, y, li, ny, L->o, acc);
        for (int u = 0; u < mo->nlive; u++) {
            int i = mo->live[u];
            h[i] = bf(h[i] + L->o[i]), hz_set(M, i, h[i]);
        }
        bar(M, c);
        if (L->nomlp) continue;  // (a layer whose MLP is empty leaves the residual as it is)
        // --- MLP: gate and up for this thread's units
        rs = rms_scale(M);
        const Mat *mg = &L->sub[GATE][t];
        matvec_n(M, mg, L->post_norm, rs, L->g, acc);
        matvec_n(M, &L->sub[UP][t], L->post_norm, rs, L->u, acc);
        int nm = 0;
        int32_t *ml = L->ml + L->fstart[t];
        for (int u = 0; u < mg->nlive; u++) {
            int i = mg->live[u];
            L->mm[i] = L->u[i] == 0.0f ? 0.0f : bf(bf(silu(L->g[i])) * L->u[i]);
            if (L->mm[i] != 0.0f) ml[nm++] = i;
        }
        L->mn[t] = nm;
        bar(M, c);
        // --- down projection, this thread's residual rows
        nm = 0;
        for (int u = 0; u < NT; u++)
            for (int p = 0; p < L->mn[u]; p++) li[nm++] = L->ml[L->fstart[u] + p];
        const Mat *md = &L->sub[DOWN][t];
        matvec(md, L->mm, li, nm, L->dn, acc);
        for (int u = 0; u < md->nlive; u++) {
            int i = md->live[u];
            h[i] = bf(h[i] + L->dn[i]), hz_set(M, i, h[i]);
        }
        bar(M, c);
    }
}

typedef struct { Model *M; int t; } Arg;

static void *worker(void *p) {
    Arg a = *(Arg *)p;
    free(p);
    Model *M = a.M;
    int seen = 0;
    for (;;) {
        int g;
        while ((g = atomic_load_explicit(&M->gen, memory_order_acquire)) == seen)
            __asm__ volatile("yield");
        seen = g;
        layers(M, a.t);
        if (M->head) {
            bar(M, &M->cx[a.t]);  // the final norm's values are in hx
            head_range(M, a.t);
        }
        atomic_fetch_add_explicit(&M->done, 1, memory_order_release);
    }
}

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
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int t = 1; t < NT; t++) {
        Arg *a = malloc(sizeof(Arg));
        a->M = M, a->t = t;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);  // the performance cores
        pthread_create(&M->th[t], &at, worker, a);
    }
}

// One token through every layer; returns the argmax of the bf16 logits (the first, on ties), or
// -1 when head is 0 (a token whose prediction is not read).
int st_step(Model *M, int tok, int head) {
    float *h = M->h;
    memset(h, 0, sizeof(float) * M->D), memset(M->hz, 0, sizeof(uint64_t) * ((M->D + 63) / 64));
    for (int p = M->erow[tok]; p < M->erow[tok + 1]; p++)
        h[M->ecol[p]] = M->eval[p], hz_set(M, M->ecol[p], M->eval[p]);
    M->head = head;
    atomic_store_explicit(&M->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&M->gen, 1, memory_order_release);
    layers(M, 0);
    if (head) {
        // the final norm (its gain is 0 outside the code dims) and the output head
        float rs = rms_scale(M);
        for (int k = 0; k < M->ncode; k++) M->hx[k] = normed(M, M->norm, rs, M->code[k]);
        bar(M, &M->cx[0]);
        head_range(M, 0);
    }
    while (atomic_load_explicit(&M->done, memory_order_acquire) < NT - 1) __asm__ volatile("yield");
    M->tick++;
    if (!head) return -1;
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
