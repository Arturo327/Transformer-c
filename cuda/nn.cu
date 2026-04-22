// nn.cu — versión CUDA de nn.c
// Cambios principales respecto a nn.c:
//	 - todos los malloc/calloc → cudaMalloc/cudaMemset (memoria en GPU)
//	 - las 4 funciones mat_mul → cublasSgemm (10-20x más rápido)
//	 - las ops element-wise → kernels CUDA
//	 - str_w_file / load_w_file usan cudaMemcpy para leer/escribir pesos

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define EPSILON 1e-8f
#define BETA1 0.9f
#define BETA2 0.999f
#define BETA1_MINUS 0.1f
#define BETA2_MINUS 0.001f

static cublasHandle_t cublas_handle;

#define CUDA_CHECK(call) do { \
	cudaError_t e = (call); \
	if (e != cudaSuccess) { \
		fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(1); \
	} \
} while(0)

#define CUBLAS_CHECK(call) do { \
	cublasStatus_t s = (call); \
	if (s != CUBLAS_STATUS_SUCCESS) { \
		fprintf(stderr, "cuBLAS error %s:%d  código %d\n", __FILE__, __LINE__, s); \
		exit(1); \
	} \
} while(0)

uint64_t xr_state = 0x853c49e6748fea9bULL;

uint64_t xorshift64(void) {
	xr_state ^= xr_state << 13;
	xr_state ^= xr_state >> 7;
	xr_state ^= xr_state << 17;
	return xr_state;
}

void softmax(float *in, float *out, int len) {
	float max = in[0];
	for (int i = 1; i < len; i++) if (in[i] > max) max = in[i];
	float sum = 0.0f;
	for (int i = 0; i < len; i++) { out[i] = expf(in[i] - max); sum += out[i]; }
	float inv = 1.0f / sum;
	for (int i = 0; i < len; i++) out[i] *= inv;
}

static float normal_random(void) {
	static int has_spare = 0;
	static float spare;
	if (has_spare) { has_spare = 0; return spare; }
	float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
	float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
	float a = sqrtf(-2.0f * logf(u1));
	float b = 2.0f * (float)M_PI * u2;
	spare = a * sinf(b);
	has_spare = 1;
	return a * cosf(b);
}

__global__ void relu_fwd_k(const float *in, float *out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = fmaxf(in[i], 0.0f);
}

__global__ void relu_bck_k(const float *out, const float *gin, float *gout, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) gout[i] += gin[i] * (out[i] > 0.0f ? 1.0f : 0.0f);
}

__global__ void add_fwd_k(const float *a, const float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] + b[i];
}

__global__ void add_bck_k(const float *gc, float *ga, float *gb, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) { ga[i] += gc[i]; gb[i] += gc[i]; }
}

__global__ void bias_fwd_k(float *Y, const float *W, int num_in, int num_out, int total) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < total) {
		int col = i % num_out;
		Y[i] += W[col * (num_in + 1) + num_in];
	}
}

// gradiente del bias (grad_W[i*(num_in+1)+num_in] += sum_s gin[s*num_out+i])
__global__ void bias_bck_k(const float *gin, float *gW, int num_in, int num_out, int seq_len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_out) {
		float s = 0.0f;
		for (int k = 0; k < seq_len; k++) s += gin[k * num_out + i];
		gW[i * (num_in + 1) + num_in] += s;
	}
}

__global__ void norm_fwd_k(const float *in, float *out, float *x_hat,
							 float *inv_var, const float *w, int n, int seq_len) {
	int pos = blockIdx.x;
	if (pos >= seq_len) return;
	const float *inp = in + pos * n;
	float *o   = out   + pos * n;
	float *xh  = x_hat + pos * n;

	float mean = 0.0f, M2 = 0.0f;
	for (int i = 0; i < n; i++) {
		float d = inp[i] - mean;
		mean += d / (i + 1);
		M2	 += d * (inp[i] - mean);
	}
	float ivar = 1.0f / sqrtf(M2 / n + 1e-8f);
	inv_var[pos] = ivar;
	for (int i = 0; i < n; i++) {
		xh[i] = (inp[i] - mean) * ivar;
		o[i]  = xh[i] * w[i * 2] + w[i * 2 + 1];
	}
}

__global__ void norm_bck_k(const float *gin, float *gout, float *gw,
							 const float *xh, const float *inv_var,
							 const float *w, int n, int seq_len) {
	int pos = blockIdx.x;
	if (pos >= seq_len) return;
	const float *gi = gin  + pos * n;
	float *go		 = gout + pos * n;
	const float *xhp = xh  + pos * n;
	float ivar = inv_var[pos];

	// dx_hat se almacena en registros locales (n <= 1024)
	float dx[1024];
	float m1 = 0.0f, m2 = 0.0f;
	for (int i = 0; i < n; i++) {
		dx[i] = gi[i] * w[i * 2];
		m1 += dx[i];
		m2 += dx[i] * xhp[i];
		atomicAdd(&gw[i * 2],	  xhp[i] * gi[i]);
		atomicAdd(&gw[i * 2 + 1], gi[i]);
	}
	m1 /= n; m2 /= n;
	for (int i = 0; i < n; i++)
		go[i] += ivar * (dx[i] - m1 - xhp[i] * m2);
}

__global__ void dropout_fwd_k(const float *in, float *out, float *mask,
								float keep_prob, float scale, int n, uint64_t seed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	uint64_t st = seed + (uint64_t)i * 6364136223846793005ULL + 1442695040888963407ULL;
	st ^= st >> 33; st *= 0xff51afd7ed558ccdULL; st ^= st >> 33;
	float u = (float)(st >> 11) * (1.0f / (float)(1ULL << 53));
	float m = (u < keep_prob) ? scale : 0.0f;
	mask[i] = m;
	out[i]	= in[i] * m;
}

__global__ void dropout_bck_k(const float *gin, float *gout, const float *mask, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) gout[i] += gin[i] * mask[i];
}

__global__ void emb_fwd_k(const float *tokens, const float *w, const float *pe,
							float *out, int dim, int seq_len) {
	int pos = blockIdx.x;
	int j	= threadIdx.x;	 // threadIdx cubre dim (máx 1024)
	if (pos < seq_len && j < dim) {
		int tok = (int)tokens[pos];
		out[pos * dim + j] = w[tok * dim + j] + pe[pos * dim + j];
	}
}

__global__ void emb_bck_k(const float *tokens, const float *gin, float *gw,
							int dim, int seq_len) {
	int pos = blockIdx.x;
	int j	= threadIdx.x;
	if (pos < seq_len && j < dim) {
		int tok = (int)tokens[pos];
		atomicAdd(&gw[tok * dim + j], gin[pos * dim + j]);
	}
}

__global__ void attn_softmax_k(float *A, int seq_len, float inv_dim) {
	int row = blockIdx.x;
	if (row >= seq_len) return;
	float *r = A + row * seq_len;
	for (int j = 0; j <= row; j++) r[j] *= inv_dim;
	for (int j = row + 1; j < seq_len; j++) r[j] = -1e9f;
	float mx = r[0];
	for (int j = 1; j <= row; j++) if (r[j] > mx) mx = r[j];
	float sum = 0.0f;
	for (int j = 0; j <= row; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
	float inv = 1.0f / sum;
	for (int j = 0; j <= row; j++) r[j] *= inv;
	for (int j = row + 1; j < seq_len; j++) r[j] = 0.0f;
}

__global__ void attn_softbck_k(const float *gA, float *gS, const float *A,
								 int seq_len, float inv_dim) {
	int row = blockIdx.x;
	if (row >= seq_len) return;
	const float *ga = gA + row * seq_len;
	float *gs		 = gS + row * seq_len;
	const float *a	 = A  + row * seq_len;
	float dot = 0.0f;
	for (int j = 0; j <= row; j++) dot += ga[j] * a[j];
	for (int j = 0; j <= row; j++) gs[j] = (ga[j] - dot) * a[j] * inv_dim;
	for (int j = row + 1; j < seq_len; j++) gs[j] = 0.0f;
}

__global__ void adam_k(float *w, float *m, float *v, float *grad,
						float lr, float b1t, float b2t,
						float lambda, int batch, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float g = grad[i] / (float)batch + lambda * w[i];
	m[i] = BETA1 * m[i] + BETA1_MINUS * g;
	v[i] = BETA2 * v[i] + BETA2_MINUS * g * g;
	float mh = m[i] / (1.0f - b1t);
	float vh = v[i] / (1.0f - b2t);
	w[i] -= lr * mh / (sqrtf(vh) + 1e-8f);
	grad[i] = 0.0f;
}

__global__ void loss_grad_k(float *logits, const int *targets, float *d_loss,
							  int vocab_size) {
	int h = blockIdx.x;
	float *row = logits + h * vocab_size;
	int target = targets[h];

	float mx = row[0];
	for (int i = 1; i < vocab_size; i++) if (row[i] > mx) mx = row[i];
	float sum = 0.0f;
	for (int i = 0; i < vocab_size; i++) { row[i] = expf(row[i] - mx); sum += row[i]; }
	float inv = 1.0f / sum;
	float p_target = row[target] * inv;
	atomicAdd(d_loss, -logf(p_target + 1e-9f));
	for (int i = 0; i < vocab_size; i++) row[i] *= inv;
	row[target] -= 1.0f;
}

void compute_loss_grad(float *d_logits, int *d_targets, float *d_loss,
						int seq_len, int vocab_size) {
	CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
	loss_grad_k<<<seq_len, 1>>>(d_logits, d_targets, d_loss, vocab_size);
	CUDA_CHECK(cudaGetLastError());
}

static inline int blocks(int n, int tpb) { return (n + tpb - 1) / tpb; }
static const float ONE	= 1.0f;
static const float ZERO = 0.0f;

// Nota sobre cuBLAS (columna-mayor) para matrices almacenadas en fila-mayor:
//	 C(m×n) = A(m×k) * B(k×n)  →  cublasSgemm(N,N, n,m,k, B,n, A,k, C,n)
//	 C(m×n) = A(m×k) * B^T	   →  cublasSgemm(T,N, n,m,k, B,k, A,k, C,n)   B es (n×k)
//	 C(m×n) = A^T * B			→  cublasSgemm(N,T, n,m,k, B,n, A,n, C,n)	A es (k×m)

// ─── Layer forward / backward ─────────────────────────────────────────────────

void relu_forward(Layer *layer) {
	int n = layer->num_out * layer->seq_len;
	relu_fwd_k<<<blocks(n,256), 256>>>(layer->inputs[0], layer->outputs, n);
}

void relu_back(Layer *layer) {
	int n = layer->num_out * layer->seq_len;
	relu_bck_k<<<blocks(n,256), 256>>>(layer->outputs, layer->grad_out,
										layer->grad_in[0], n);
}

void add_forward(Layer *layer) {
	int n = layer->num_out * layer->seq_len;
	add_fwd_k<<<blocks(n,256), 256>>>(layer->inputs[0], layer->inputs[1],
									   layer->outputs, n);
}

void add_back(Layer *layer) {
	int n = layer->num_out * layer->seq_len;
	add_bck_k<<<blocks(n,256), 256>>>(layer->grad_out, layer->grad_in[0],
									   layer->grad_in[1], n);
}

void embedding_forward(Layer *layer) {
	// un bloque por posición, un hilo por dimensión
	emb_fwd_k<<<layer->seq_len, layer->num_out>>>(
		layer->inputs[0], layer->w, layer->inv_var,
		layer->outputs, layer->num_out, layer->seq_len);
}

void embedding_back(Layer *layer) {
	emb_bck_k<<<layer->seq_len, layer->num_out>>>(
		layer->inputs[0], layer->grad_out, layer->grad,
		layer->num_out, layer->seq_len);
}

void FCforward(Layer *layer) {
	int ni = layer->num_in, no = layer->num_out, sl = layer->seq_len;
	CUBLAS_CHECK(cublasSgemm(cublas_handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		no, sl, ni,
		&ONE, layer->w, ni + 1,
			  layer->inputs[0], ni,
		&ZERO, layer->outputs, no));
	bias_fwd_k<<<blocks(sl*no, 256), 256>>>(layer->outputs, layer->w, ni, no, sl*no);
}

void FCback(Layer *layer) {
	int ni = layer->num_in, no = layer->num_out, sl = layer->seq_len;
	CUBLAS_CHECK(cublasSgemm(cublas_handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		ni, sl, no,
		&ONE, layer->w, ni + 1,
			  layer->grad_out, no,
		&ONE, layer->grad_in[0], ni));
	CUBLAS_CHECK(cublasSgemm(cublas_handle,
		CUBLAS_OP_N, CUBLAS_OP_T,
		ni, no, sl,
		&ONE, layer->inputs[0], ni,
			  layer->grad_out, no,
		&ONE, layer->grad, ni + 1));

	bias_bck_k<<<blocks(no,256), 256>>>(layer->grad_out, layer->grad, ni, no, sl);
}

void norm_forward(Layer *layer) {
	norm_fwd_k<<<layer->seq_len, 1>>>(
		layer->inputs[0], layer->outputs, layer->x_hat,
		layer->inv_var, layer->w,
		layer->num_out, layer->seq_len);
}

void norm_back(Layer *layer) {
	norm_bck_k<<<layer->seq_len, 1>>>(
		layer->grad_out, layer->grad_in[0], layer->grad,
		layer->x_hat, layer->inv_var, layer->w,
		layer->num_out, layer->seq_len);
}

void dropout_forward(Layer *layer) {
	int n = layer->num_in * layer->seq_len;
	if (!layer->training) {
		CUDA_CHECK(cudaMemcpy(layer->outputs, layer->inputs[0],
							  n * sizeof(float), cudaMemcpyDeviceToDevice));
		return;
	}
	float scale		= layer->inv_std;
	float keep_prob = 1.0f - 1.0f / scale;
	uint64_t seed	= xorshift64();
	dropout_fwd_k<<<blocks(n,256), 256>>>(
		layer->inputs[0], layer->outputs, layer->x_hat,
		keep_prob, scale, n, seed);
}

void dropout_back(Layer *layer) {
	int n = layer->num_in * layer->seq_len;
	dropout_bck_k<<<blocks(n,256), 256>>>(
		layer->grad_out, layer->grad_in[0], layer->x_hat, n);
}

void attention_forward(Layer *layer) {
	const int dim  = layer->num_in;
	const int sl   = layer->seq_len;
	const int H    = (int)layer->inv_std;
	const int dh   = dim / H;
	const float inv_dim = 1.0f / sqrtf((float)dh);

	float *Q   = layer->inputs[0];
	float *K   = layer->inputs[1];
	float *V   = layer->inputs[2];
	float *A   = layer->x_hat;
	float *out = layer->outputs;
	CUDA_CHECK(cudaMemset(out, 0, dim * sl * sizeof(float)));

	for (int h = 0; h < H; h++) {
		float *Qh  = Q + h * dh;
		float *Kh  = K + h * dh;
		float *Vh  = V + h * dh;
		float *Ah  = A + h * sl * sl;
		float *outh = out + h * dh;

		CUBLAS_CHECK(cublasSgemm(cublas_handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			sl, sl, dh,
			&ONE, Kh, dim, Qh, dim, &ZERO, Ah, sl));

		attn_softmax_k<<<sl, 1>>>(Ah, sl, inv_dim);

		CUBLAS_CHECK(cublasSgemm(cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			dh, sl, sl,
			&ONE, Vh, dim, Ah, sl, &ZERO, outh, dim));
	}
}

void attention_back(Layer *layer) {
	const int dim  = layer->num_in;
	const int sl   = layer->seq_len;
	const int H    = (int)layer->inv_std;
	const int dh   = dim / H;
	const float inv_dim = 1.0f / sqrtf((float)dh);

	float *Q	 = layer->inputs[0];
	float *K	 = layer->inputs[1];
	float *V	 = layer->inputs[2];
	float *A	 = layer->x_hat;
	float *gQ	 = layer->grad_in[0];
	float *gK	 = layer->grad_in[1];
	float *gV	 = layer->grad_in[2];
	float *gout  = layer->grad_out;
	float *gA	 = layer->w;
	float *gS	 = layer->m; 

	for (int h = 0; h < H; h++) {
		float *Qh	= Q    + h * dh;
		float *Kh	= K    + h * dh;
		float *Vh	= V    + h * dh;
		float *Ah	= A    + h * sl * sl;
		float *gQh	= gQ   + h * dh;
		float *gKh	= gK   + h * dh;
		float *gVh	= gV   + h * dh;
		float *gouth = gout + h * dh;

		CUBLAS_CHECK(cublasSgemm(cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			sl, sl, dh,
			&ONE, Vh, dim, gouth, dim, &ZERO, gA, sl));

		attn_softbck_k<<<sl, 1>>>(gA, gS, Ah, sl, inv_dim);

		CUBLAS_CHECK(cublasSgemm(cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			dh, sl, sl,
			&ONE, Kh, dim, gS, sl, &ONE, gQh, dim));

		CUBLAS_CHECK(cublasSgemm(cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			dh, sl, sl,
			&ONE, Qh, dim, gS, sl, &ONE, gKh, dim));

		CUBLAS_CHECK(cublasSgemm(cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_T,
			dh, sl, sl,
			&ONE, gouth, dim, Ah, sl, &ONE, gVh, dim));
	}
}

void forward(Network *net, float *d_input, float *d_output) {
	Layer *layer = net->layers[0];
	int a = layer->num_in * layer->seq_len;
	CUDA_CHECK(cudaMemcpy(layer->outputs, d_input, a * sizeof(float),
						  cudaMemcpyDeviceToDevice));
	for (int i = 1; i < net->num_layers; i++)
		net->layers[i]->forward(net->layers[i]);
	layer = net->layers[net->num_layers - 1];
	a = layer->seq_len * layer->num_out;
	CUDA_CHECK(cudaMemcpy(d_output, layer->outputs, a * sizeof(float),
						  cudaMemcpyDeviceToDevice));
}

void backward(Network *net, float *d_error) {
	for (int i = 0; i < net->num_layers - 1; i++) {
		Layer *l = net->layers[i];
		CUDA_CHECK(cudaMemset(l->grad_out, 0,
							  l->num_out * l->seq_len * sizeof(float)));
	}
	Layer *last = net->layers[net->num_layers - 1];
	int a = last->num_out * last->seq_len;
	CUDA_CHECK(cudaMemcpy(last->grad_out, d_error, a * sizeof(float),
						  cudaMemcpyDeviceToDevice));
	last->backward(last);
	for (int i = net->num_layers - 2; i > 0; i--)
		net->layers[i]->backward(net->layers[i]);
}

void actualizar(Network *net, float lr, int batch_size, float lambda) {
	float b1t = net->beta1_t, b2t = net->beta2_t;
	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = net->layers[i];
		int n = l->total_w;
		if (n == 0) continue;
		float llr = (l->type == 9) ? lr * 0.6f : lr;
		adam_k<<<blocks(n,256), 256>>>(l->w, l->m, l->v, l->grad,
										llr, b1t, b2t, lambda, batch_size, n);
	}
	net->beta1_t *= BETA1;
	net->beta2_t *= BETA2;
	CUDA_CHECK(cudaDeviceSynchronize());
}

void iniciar_w(Layer *layer) {
	int n = layer->total_w;
	float *h = (float*)malloc(n * sizeof(float));
	for (int i = 0; i < n; i++) h[i] = normal_random() * 0.02f;
	CUDA_CHECK(cudaMemcpy(layer->w, h, n * sizeof(float), cudaMemcpyHostToDevice));
	free(h);
}

int str_w_file(Network *net, const char *filename) {
	FILE *f = fopen(filename, "wb");
	if (!f) return 0;
	fwrite(&net->beta1_t, sizeof(float), 1, f);
	fwrite(&net->beta2_t, sizeof(float), 1, f);
	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = net->layers[i];
		int sz = l->total_w;
		if (sz == 0) continue;
		float *h = (float*)malloc(sz * sizeof(float));
		CUDA_CHECK(cudaMemcpy(h, l->w, sz*sizeof(float), cudaMemcpyDeviceToHost));
		fwrite(h, sizeof(float), sz, f);
		CUDA_CHECK(cudaMemcpy(h, l->v, sz*sizeof(float), cudaMemcpyDeviceToHost));
		fwrite(h, sizeof(float), sz, f);
		CUDA_CHECK(cudaMemcpy(h, l->m, sz*sizeof(float), cudaMemcpyDeviceToHost));
		fwrite(h, sizeof(float), sz, f);
		free(h);
	}
	fclose(f);
	return 1;
}

int load_w_file(Network *net, const char *filename) {
	FILE *f = fopen(filename, "rb");
	if (!f) return 0;
	int a = fread(&net->beta1_t, sizeof(float), 1, f);
	a = fread(&net->beta2_t, sizeof(float), 1, f);
	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = net->layers[i];
		int sz = l->total_w;
		if (sz == 0) continue;
		float *h = (float*)malloc(sz * sizeof(float));
		a = fread(h, sizeof(float), sz, f);
		CUDA_CHECK(cudaMemcpy(l->w, h, sz*sizeof(float), cudaMemcpyHostToDevice));
		a = fread(h, sizeof(float), sz, f);
		CUDA_CHECK(cudaMemcpy(l->v, h, sz*sizeof(float), cudaMemcpyHostToDevice));
		a = fread(h, sizeof(float), sz, f);
		CUDA_CHECK(cudaMemcpy(l->m, h, sz*sizeof(float), cudaMemcpyHostToDevice));
		free(h);
	}
	fclose(f);
	return 1;
}


static void* gpu_alloc(size_t bytes) {
	void *ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, bytes));
	return ptr;
}
static void* gpu_calloc(size_t n, size_t elem) {
	void *ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, n * elem));
	CUDA_CHECK(cudaMemset(ptr, 0, n * elem));
	return ptr;
}
#define GALLOC(n) (float*) gpu_alloc(sizeof(float)*(n))
#define GCALLOC(n) (float*) gpu_calloc(n, sizeof(float))

Network *defnn(int num_layers, Layer **layers) {
	CUBLAS_CHECK(cublasCreate(&cublas_handle));
	Network *net = (Network*)malloc(sizeof(Network));
	net->num_layers = num_layers;
	net->layers = (Layer**)malloc(sizeof(Layer*) * num_layers);
	for (int i = 0; i < num_layers; i++) net->layers[i] = layers[i];
	net->beta1_t  = BETA1;
	net->beta2_t  = BETA2;
	net->forward  = forward;
	net->backward = backward;
	net->actualizar = actualizar;
	return net;
}

Layer *defL_FC(int num_in, int num_out, int seq_len, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 1; l->num_in = num_in; l->num_out = num_out;
	l->total_w = num_out * (num_in + 1);
	l->training = 1; l->seq_len = seq_len; l->inv_var = NULL;
	l->inputs  = (float**)malloc(sizeof(float*));
	l->grad_in = (float**)malloc(sizeof(float*));
	l->inputs[0]  = layers[0]->outputs;
	l->grad_in[0] = layers[0]->grad_out;
	l->w	= GALLOC(l->total_w);
	l->v	= GCALLOC(l->total_w);
	l->m	= GCALLOC(l->total_w);
	l->grad = GCALLOC(l->total_w);
	l->outputs	= GALLOC(num_out * seq_len);
	l->grad_out = GCALLOC(num_out * seq_len);
	l->x_hat = NULL; l->inv_std = 0;
	l->forward	= FCforward;
	l->backward = FCback;
	return l;
}

Layer *defL_relu(int num_in, int seq_len, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 2; l->num_in = num_in; l->num_out = num_in;
	l->total_w = 0; l->training = 1; l->seq_len = seq_len; l->inv_var = NULL;
	l->inputs  = (float**)malloc(sizeof(float*));
	l->grad_in = (float**)malloc(sizeof(float*));
	l->inputs[0]  = layers[0]->outputs;
	l->grad_in[0] = layers[0]->grad_out;
	l->w = l->v = l->m = l->grad = NULL;
	l->outputs	= GALLOC(num_in * seq_len);
	l->grad_out = GCALLOC(num_in * seq_len);
	l->x_hat = NULL; l->inv_std = 0;
	l->forward	= relu_forward;
	l->backward = relu_back;
	return l;
}

Layer *defL_norm(int num_in, int seq_len, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 6; l->num_in = num_in; l->num_out = num_in;
	int tw = 2 * num_in;
	l->total_w = tw; l->training = 1; l->seq_len = seq_len;
	l->inputs  = (float**)malloc(sizeof(float*));
	l->grad_in = (float**)malloc(sizeof(float*));
	l->inputs[0]  = layers[0]->outputs;
	l->grad_in[0] = layers[0]->grad_out;
	l->grad = GCALLOC(tw);
	// inicializar gamma=1, beta=0 en CPU y copiar
	float *h = (float*)malloc(tw * sizeof(float));
	for (int i = 0; i < tw; i += 2) { h[i] = 1.0f; h[i+1] = 0.0f; }
	l->w = GALLOC(tw);
	CUDA_CHECK(cudaMemcpy(l->w, h, tw*sizeof(float), cudaMemcpyHostToDevice));
	free(h);
	l->v = GCALLOC(tw);
	l->m = GCALLOC(tw);
	l->outputs	= GALLOC(num_in * seq_len);
	l->grad_out = GCALLOC(num_in * seq_len);
	l->x_hat	= GALLOC(num_in * seq_len);
	l->inv_var	= GALLOC(seq_len);
	l->inv_std	= 0;
	l->forward	= norm_forward;
	l->backward = norm_back;
	return l;
}

Layer *defL_add(int num_out, int seq_len, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 5; l->num_in = num_out; l->num_out = num_out;
	l->total_w = 0; l->training = 1; l->seq_len = seq_len; l->inv_var = NULL;
	l->w = l->v = l->m = l->grad = NULL;
	l->inputs  = (float**)malloc(sizeof(float*) * 2);
	l->grad_in = (float**)malloc(sizeof(float*) * 2);
	l->inputs[0]  = layers[0]->outputs;  l->inputs[1]  = layers[1]->outputs;
	l->grad_in[0] = layers[0]->grad_out; l->grad_in[1] = layers[1]->grad_out;
	l->outputs	= GALLOC(num_out * seq_len);
	l->grad_out = GCALLOC(num_out * seq_len);
	l->x_hat = NULL; l->inv_std = 0;
	l->forward	= add_forward;
	l->backward = add_back;
	return l;
}

Layer *defL_input(int num_in, int seq_len) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 4; l->num_in = num_in; l->num_out = num_in;
	l->total_w = 0; l->training = 1; l->seq_len = seq_len; l->inv_var = NULL;
	l->w = l->v = l->m = l->grad = NULL;
	l->inputs = l->grad_in = NULL;
	l->outputs	= GALLOC(num_in * seq_len);
	l->grad_out = GCALLOC(num_in * seq_len);
	l->x_hat = NULL; l->inv_std = 0;
	l->forward = l->backward = NULL;
	return l;
}

Layer *defL_dropout(int num_in, float p, int seq_len, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 7; l->training = 1; l->num_in = num_in; l->num_out = num_in;
	l->total_w = 0; l->seq_len = seq_len; l->inv_var = NULL;
	l->inv_std = 1.0f / (1.0f - p);
	l->inputs  = (float**)malloc(sizeof(float*));
	l->grad_in = (float**)malloc(sizeof(float*));
	l->inputs[0]  = layers[0]->outputs;
	l->grad_in[0] = layers[0]->grad_out;
	l->outputs	= GALLOC(num_in * seq_len);
	l->grad_out = GCALLOC(num_in * seq_len);
	l->x_hat	= GALLOC(num_in * seq_len);  // máscara
	l->w = l->v = l->m = l->grad = NULL;
	l->forward	= dropout_forward;
	l->backward = dropout_back;
	return l;
}

Layer *defL_attention(int num_in, int seq_len, int heads, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 8; l->training = 1; l->num_in = num_in; l->num_out = num_in;
	l->total_w = 0; l->inv_var = NULL;
	l->seq_len = seq_len; l->inv_std = (float)heads;
	l->inputs  = (float**)malloc(sizeof(float*) * 3);
	l->grad_in = (float**)malloc(sizeof(float*) * 3);
	for (int i = 0; i < 3; i++) {
		l->inputs[i]  = layers[i]->outputs;
		l->grad_in[i] = layers[i]->grad_out;
	}
	l->outputs	= GALLOC(num_in * seq_len);
	l->grad_out = GCALLOC(num_in * seq_len);
	l->x_hat	= GALLOC(heads * seq_len * seq_len);  // A por cabeza
	l->w		= GALLOC(seq_len * seq_len);			// gA buffer
	l->m		= GALLOC(seq_len * seq_len);			// gS buffer
	l->v		= GALLOC(seq_len * num_in);				// tmp
	l->grad		= NULL;
	l->forward	= attention_forward;
	l->backward = attention_back;
	return l;
}

Layer *defL_embedding(int vocab_size, int dim, int seq_len, Layer **layers) {
	Layer *l = (Layer*)malloc(sizeof(Layer));
	l->type = 9; l->num_in = 1; l->num_out = dim;
	l->total_w = dim * vocab_size;
	l->training = 1; l->seq_len = seq_len; l->inv_std = 0; l->x_hat = NULL;

	l->inputs  = (float**)malloc(sizeof(float*));
	l->grad_in = (float**)malloc(sizeof(float*));
	l->inputs[0]  = layers[0]->outputs;
	l->grad_in[0] = layers[0]->grad_out;

	l->w	= GALLOC(vocab_size * dim);
	l->v	= GCALLOC(vocab_size * dim);
	l->m	= GCALLOC(vocab_size * dim);
	l->grad = GCALLOC(vocab_size * dim);
	l->outputs	= GALLOC(dim * seq_len);
	l->grad_out = GCALLOC(dim * seq_len);

	l->inv_var = GALLOC(seq_len * dim);
	float *h_pe = (float*)malloc(seq_len * dim * sizeof(float));
	for (int i = 0; i < seq_len; i++) {
		for (int j = 0; j < dim; j++) {
			float angle = i / powf(10000.0f, (2*(j/2)) / (float)dim);
			h_pe[i*dim+j] = (j%2 == 0) ? sinf(angle) : cosf(angle);
		}
	}
	CUDA_CHECK(cudaMemcpy(l->inv_var, h_pe, seq_len*dim*sizeof(float),
						  cudaMemcpyHostToDevice));
	free(h_pe);

	l->forward	= embedding_forward;
	l->backward = embedding_back;
	return l;
}

void net_set_training(Network *net, int training) {
	for (int i = 0; i < net->num_layers; i++)
		net->layers[i]->training = training;
}

void net_set_seq_len(Network *net, int seq_len) {
	for (int i = 0; i < net->num_layers; i++)
		net->layers[i]->seq_len = seq_len;
}

void freeL(Layer *layer) {
	cudaFree(layer->w);    cudaFree(layer->v);
	cudaFree(layer->m);    cudaFree(layer->grad);
	cudaFree(layer->outputs); cudaFree(layer->grad_out);
	cudaFree(layer->x_hat);   cudaFree(layer->inv_var);
	free(layer->inputs);   free(layer->grad_in);
	free(layer);
}

void freeN(Network *net) {
	for (int i = 0; i < net->num_layers; i++) freeL(net->layers[i]);
	free(net->layers);
	free(net);
	cublasDestroy(cublas_handle);
}

int *translate(char *input, char **vocab, Merge *merges, int num_merges,
			   int max_token_len, int vocab_size, int *out_len) {
	int len = strlen(input);
	int *tokens = (int*)malloc(sizeof(int) * len);
	int n = 0;
	for (int i = 0; i < len; i++) {
		char s[2] = {input[i], '\0'};
		int found = -1;
		for (int j = 0; j < vocab_size; j++)
			if (strcmp(vocab[j], s) == 0) { found = j; break; }
		tokens[n++] = (found == -1) ? 0 : found;
	}
	for (int m = 0; m < num_merges; m++) {
		int a = merges[m].a, b = merges[m].b, r = merges[m].result;
		int new_n = 0;
		for (int i = 0; i < n; ) {
			if (i+1 < n && tokens[i] == a && tokens[i+1] == b) {
				tokens[new_n++] = r; i += 2;
			} else {
				tokens[new_n++] = tokens[i++];
			}
		}
		n = new_n;
	}
	int *out = (int*)malloc(sizeof(int) * n);
	memcpy(out, tokens, n * sizeof(int));
	free(tokens);
	*out_len = n;
	return out;
}
