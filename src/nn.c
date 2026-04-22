#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define EPSILON 1e-8f
#define BETA1 0.9f
#define BETA2 0.999f
#define BETA1_MINUS 0.1f
#define BETA2_MINUS 0.001f

static inline float sigmoid (float a) {return 1.0f / (1.0f + expf(-a)); }

static inline float d_sigmoid (float a) {return a * (1.0f - a); }

static inline float relu (float a) { return fmax(a, 0.0f); }

static inline float d_relu (float a) { return (float)(a > 0.0f); }

uint64_t xr_state = 0x853c49e6748fea9bULL;

uint64_t xorshift64(void) {
	xr_state ^= xr_state << 13;
	xr_state ^= xr_state >> 7;
	xr_state ^= xr_state << 17;
	return xr_state;
}

void softmax (float *in, float *out, int len) {
	float max = in[0];
        for (int i = 1; i < len; i++) {
                if (in[i] > max) {
                        max = in[i];
                }
        }
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
                out[i] = expf(in[i] - max);
                sum += out[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < len; i++) {
                out[i] *= inv_sum;
        }
        return;
}

void soft_back (float *grad_in, float *grad_out, float *out, int len) {
	float a = 0.0f;
        for (int i = 0; i < len; i++) {
                a += grad_in[i] * out[i];
        }
        for (int i = 0; i < len; i++) {
                grad_out[i] += (grad_in[i] - a) * out[i];
        }
        return;
}

void mat_mul (float * restrict a, float * restrict b, int len, int lenA, int lenB, int offsetA, int offsetB, int offsetR, float * restrict result) {  // A x B
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < lenA; i++) {
		float *row_r = result + offsetR * i;
		float *row_a = a + offsetA * i;
		for (int j = 0; j < lenB; j++) {
			row_r[j] = 0.0f;
		}
        	for (int k = 0; k < len; k++) {
            		float aik = row_a[k];
			float *row_b = b + offsetB * k;
            		for (int j = 0; j < lenB; j++) {
                		row_r[j] += aik * row_b[j]; 
            		}
        	}
    	}
}

void mat_mul_at (float * restrict a, float * restrict b, int len, int lenA, int lenB, int offsetA, int offsetB, float * restrict result) {  // A transpuesta x B
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < len; i++) {
        	float *row_r = result + i * lenB;
		for (int j = 0; j < lenB; j++) {
			row_r[j] = 0.0f;
		}
        	for (int k = 0; k < lenA; k++) {
            		float aT = a[k * offsetA + i];
			float *row_b = b + k * offsetB;
            		for (int j = 0; j < lenB; j++) {
                		row_r[j] += aT * row_b[j];
            		}
        	}
    	}
}	

void mat_mul_t (float * restrict a, float * restrict b, int len_x, int len_y, int offset, float * restrict result) {  //A x B transpuesta
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < len_y; i++) {
		float *row_r = result + i * len_y;
		float *row_a = a + i * offset;
		for (int j = 0; j < len_y; j++) {
			float sum = 0.0f;
			float *row_b = b + j * offset;
			for (int k = 0; k < len_x; k++) {
				sum += row_a[k] * row_b[k];
			}
			row_r[j] = sum;
		}
	}
}

void mat_mul_t_mask (float * restrict a, float * restrict b, int len_x, int len_y, int offset, float * restrict result) { // A x B transpuesta, no opera cuando j > i
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < len_y; i++) {
		float *row_r = result + i * len_y;
		float *row_a = a + i * offset;
		for (int j = 0; j <= i; j++) {
			float sum = 0.0f;
			float *row_b = b + j * offset;
			for (int k = 0; k < len_x; k++) {
				sum += row_a[k] * row_b[k];
			}
			row_r[j] = sum;
		}
	}
}

float normal_random (void) {
	static int has_spare = 0;
	static float spare;
	if (has_spare) {
		has_spare = 0;
		return spare;
	}
	float u1, u2, a, b;

	u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
	u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
	a = sqrtf(-2.0f * logf(u1));
	b = 2.0f * M_PI * u2;
	spare = a * sinf(b);
	has_spare = 1;
	return a * cosf(b);
}

void relu_forward (Layer *layer) {
	float * in = layer->inputs[0];
	float *out = layer->outputs;
	int a = layer->num_out * layer->seq_len;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < a; i++) {
		out[i] = relu(in[i]);
	}
	return;
}

void relu_back (Layer *layer) {
	float * grad_out = layer->grad_in[0];
	float * grad_in = layer->grad_out;
	const float * out = layer->outputs;
	int a = layer->num_out * layer->seq_len;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < a; i++) {
		grad_out[i] += grad_in[i] * d_relu(out[i]);
	}
	return;
}

void sigmoid_forward (Layer *layer) {
	float * in = layer->inputs[0];
	float * out = layer->outputs;
	int a = layer->num_out * layer->seq_len;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < a; i++) {
		out[i] = sigmoid(in[i]);
	}
	return;
}

void sigmoid_back (Layer *layer) {
	float * grad_out = layer->grad_in[0];
	float * grad_in = layer->grad_out;
	const float * out = layer->outputs;
	int a = layer->num_out * layer->seq_len;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < a; i++) {
		grad_out[i] += grad_in[i] * d_sigmoid(out[i]);
	}
	return;
}

void softmax_forward(Layer *layer) {
	int n = layer->num_out;
	float *in = layer->inputs[0];
	float *out = layer->outputs;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < layer->seq_len; i++) {
		softmax(in + i * n, out + i * n, n);
	}
	return;
}

void softmax_back (Layer *layer) {
	float * grad_out = layer->grad_in[0];
	float * grad_in = layer->grad_out;
	int n = layer->num_out;
	float * out = layer->outputs;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < layer->seq_len; i++) {
		soft_back(grad_in + i * n, grad_out + i* n, out + i * n, n);
	}
	return;
}

void add_forward (Layer *layer) {
	float *vector1 = layer->inputs[0];
	float *vector2 = layer->inputs[1];
	float *result = layer->outputs;
	int a = layer->num_out * layer->seq_len;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < a; i++) {
		result[i] = vector1[i] + vector2[i];
	}
	return;
}

void add_back (Layer *layer) {
	float *vector1 = layer->grad_in[0];
	float *vector2 = layer->grad_in[1];
	float *result = layer->grad_out;
	int a = layer->num_out * layer->seq_len;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < a; i++) {
		float g = result[i];
		vector1[i] += g;
		vector2[i] += g;
	}
}

void norm_forward (Layer *layer) {
	int n = layer->num_out;
	int seq_len = layer->seq_len;
	float *in = layer->inputs[0];
	float *out = layer->outputs;
	float *x_hat = layer->x_hat;
	float *w = layer->w;
	
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < layer->seq_len; j++) {
		float *inp_j = in + j * n;
		float *out_j = out + j * n;
		float *xhat_j = x_hat + j * n;

		float mean = 0.0f, M2 = 0.0f;
		for (int i = 0; i < n; i++) {
    			float delta = inp_j[i] - mean;
    			mean += delta / (i + 1);
    			M2 += delta * (inp_j[i] - mean);
		}

		float inv_std = 1.0f / sqrtf(M2 / n + EPSILON);
		layer->inv_var[j] = inv_std;

		for (int i = 0; i < n; i++) {
			xhat_j[i] = (inp_j[i] - mean) * inv_std;
			out_j[i] = xhat_j[i] * w[i * 2] + w[i * 2 + 1];
		}
	}
}

void norm_back (Layer *layer) {
	int n = layer->num_out;
	int seq_len = layer->seq_len;
	float *grad_out = layer->grad_in[0];
	float *grad_in = layer->grad_out;
	float *grad = layer->grad;
	float *x_hat = layer->x_hat;
	float *w = layer->w;
	float *inv_var = layer->inv_var;

	#pragma omp parallel
	{
		float dx_hat[n];
		float local_grad[2 * n];
		memset(local_grad, 0, 2 * n * sizeof(float));

		#pragma omp for schedule(static)
		for (int k = 0; k < seq_len; k++) {
			float *gin_k = grad_in + k * n;
			float *gout_k = grad_out + k * n;
			float *xhat_k = x_hat + k * n;
			float mean1 = 0.0f, mean2 = 0.0f;
			
			for (int i = 0; i < n; i++) {
				dx_hat[i] = gin_k[i] * w[i * 2];
				mean1 += dx_hat[i];
				mean2 += dx_hat[i] * xhat_k[i];
				local_grad[i * 2] += xhat_k[i] * gin_k[i];
				local_grad[i * 2 + 1] += gin_k[i];
			}
			mean1 /= n;
			mean2 /= n;
			float ivar = inv_var[k];
			for (int i = 0; i < n; i++) {
				gout_k[i] += ivar * (dx_hat[i] - mean1 - xhat_k[i] * mean2);
			}
		}
		#pragma omp critical
		for (int i = 0; i < 2 * n; i++) {
			grad[i] += local_grad[i];
		}
	}
}

void dropout_forward(Layer *layer) {
	int a = layer->seq_len * layer->num_in;
	float * restrict in = layer->inputs[0];
	float * restrict out = layer->outputs;
	float * restrict mask = layer->x_hat;
	float scale = layer->inv_std;
	float mean = 1.0f - 1.0f / scale;

	if (!layer->training) {
		memcpy(out, in, a * sizeof(float));
        	return;
	}

	uint64_t threshold = (uint64_t)(mean * (float)UINT64_MAX);

	for (int i = 0; i < a; i++) {
       		int keep = (xorshift64() > threshold);
        	mask[i] = keep ? scale : 0.0f;
		*out++ = (*in++) * mask[i];
	}
}

void dropout_back (Layer *layer) {
	int n = layer->num_in * layer->seq_len;
	float * restrict grad_out = layer->grad_in[0];
	float * restrict grad_in = layer->grad_out;
	float * restrict mask = layer->x_hat;
	for (int i = 0; i < n; i++) {
		*grad_out++ += (*grad_in++) * (*mask++);
	}
}

void attention_forward (Layer *layer) {
	const int dim = layer->num_in;
	const int seq_len = layer->seq_len;
	int heads = (int)layer->inv_std;
	int dim_head = layer->num_in / heads;
	const float inv_dim = 1 / sqrtf((float)dim_head);

	float *Q = layer->inputs[0];
	float *K = layer->inputs[1];
	float *V = layer->inputs[2];
	float *A = layer->x_hat;
	float *out = layer->outputs;
	memset(out, 0, dim * seq_len * sizeof(float));
	for (int k = 0; k < heads; k++) {
		float *fila = A;
		mat_mul_t_mask(Q, K, dim_head, seq_len, dim, A);
		for (int i = 0; i < seq_len; i++) {
			int j;
			for (j = 0; j <= i; j++) {
				fila[j] *= inv_dim;
			}
			for (; j < seq_len; j++) {
				fila[j] = -1e9f; 
			}
			softmax(fila, fila, seq_len);
			fila += seq_len;
		}
		mat_mul(A, V, seq_len, seq_len, dim_head, seq_len, dim, dim, out);
		Q += dim_head;
		K += dim_head;
		V += dim_head;
		A += seq_len * seq_len;
		out += dim_head;
	}
}

void attention_back (Layer *layer) {
	const int dim = layer->num_in;
	const int seq_len = layer->seq_len;
	int heads = (int)layer->inv_std;
	int dim_head = layer->num_in / heads;
	const float inv_dim = 1 / sqrtf((float)dim_head);

	float *Q = layer->inputs[0];
	float *K = layer->inputs[1];
	float *V = layer->inputs[2];
	float *A = layer->x_hat;
	float *grad_q = layer->grad_in[0];
	float *grad_k = layer->grad_in[1];
	float *grad_v = layer->grad_in[2];
	float *grad_out = layer->grad_out;

	float *grad_A = layer->w;  // seq_len * seq_len
	float *grad_S = layer->m;  // seq_len * seq_len
	float *tmp = layer->v;     // seq_len * dim

	int fila;
	memset(grad_A, 0, seq_len * seq_len * sizeof(float));
	memset(tmp, 0, seq_len * dim * sizeof(float));
	for (int j = 0; j < heads; j++) {
		fila = 0;
		memset(grad_S, 0, seq_len * seq_len * sizeof(float));
		mat_mul_t(grad_out, V, dim_head, seq_len, dim, grad_A);
		for (int i = 0; i < seq_len; i++) {
			soft_back(grad_A + fila, grad_S + fila, A + fila, seq_len);
			fila += seq_len;
		}

		fila = seq_len * seq_len;
		for (int i = 0; i < fila; i++) {
			grad_S[i] *= inv_dim;
		}
		
		mat_mul(grad_S, K, seq_len, seq_len, dim_head, seq_len, dim, dim_head, tmp);
		int a = 0;
		float *gq_start = grad_q;
		for (int i = 0; i < seq_len; i++) {
			for (int j = 0; j < dim_head; j++) {
				grad_q[j] += tmp[a];
				a++;
			}
			grad_q += dim;
		}
		grad_q = gq_start + dim_head;

		mat_mul_at(grad_S, Q, seq_len, seq_len, dim_head, seq_len, dim, tmp);
		a = 0;
		float *gk_start = grad_k;
		for (int i = 0; i < seq_len; i++) {
			for (int j = 0; j < dim_head; j++) {
				grad_k[j] += tmp[a];
				a++;
			}
			grad_k += dim;
		}
		grad_k = gk_start + dim_head;

		mat_mul_at(A, grad_out, seq_len, seq_len, dim_head, seq_len, dim, tmp);
		a = 0;
		float *gv_start = grad_v;
		for (int i = 0; i < seq_len; i++) {
			for (int j = 0; j < dim_head; j++) {
				grad_v[j] += tmp[a];
				a++;
			}
			grad_v += dim;
		}
		grad_v = gv_start + dim_head;

		grad_out += dim_head;
		Q += dim_head;
		V += dim_head;
		K += dim_head;
		A += seq_len * seq_len;
	}
}

void embedding_forward (Layer *layer) {
	int num_out = layer->num_out;
	int seq_len = layer->seq_len;
	float * restrict inputs = layer->inputs[0];
	float * restrict outputs = layer->outputs;
	float *start = layer->w;
	float *w;
	float * restrict pe = layer->inv_var; 
	for (int i = 0; i < seq_len; i++) {
		w = start + ((int)(*inputs++)) * num_out;
		for (int j = 0; j < num_out; j++) {
			*outputs++ = (*w++) + (*pe++);
		}
	}
}

void embedding_back (Layer *layer) {
	int num_out = layer->num_out;
	int seq_len = layer->seq_len;
	float * restrict inputs = layer->inputs[0];
	float * restrict grad_in = layer->grad_out;
	float *start = layer->grad;
	float *grad;
	for (int i = 0; i < seq_len; i++) {
		grad = start + ((int)(*inputs++)) * num_out;
		for (int j = 0; j < num_out; j++) {
			*grad++ += *grad_in++;
		}
	}
}

void FCforward (Layer *layer) {
	int num_in = layer->num_in;
	int num_out = layer->num_out;
	float *w = layer->w;
	float *in = layer->inputs[0];
	float *out = layer->outputs;
	#pragma omp parallel for schedule(static)
	for (int k = 0; k < layer->seq_len; k++) {
		float *in_k = in + k * num_in;
		float *out_k = out + k * num_out;
		for (int i = 0; i < num_out; i++) {
			float *wi = w + i * (num_in + 1);
			float z = wi[num_in];
			for (int j = 0; j < num_in; j++) {
				z += wi[j] * in_k[j];
			}
			out_k[i] = z;
		}
	}
	return;
}

void FCback (Layer *layer) {
	int num_in = layer->num_in;
	int num_out = layer->num_out;
	int seq_len = layer->seq_len;
	float *grad_out = layer->grad_in[0];
	float *grad_in = layer->grad_out;
	float *grad = layer->grad;
	float *inputs = layer->inputs[0];
	float *w = layer->w;
	
	#pragma omp parallel for schedule(static)
	for (int k = 0; k < seq_len; k++) {
		float *gin_k = grad_in + k * num_out;
		float *gout_k = grad_out + k * num_in;
		for (int j = 0; j < num_in; j++) {
			float s = 0.0f;
			for (int i = 0; i < num_out; i++) {
				s += gin_k[i] * w[i * (num_in + 1) + j];
			}
			gout_k[j] += s;
		}
	}
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < num_out; i++) {
		float *gi = grad + i * (num_in + 1);
		for (int k = 0; k < seq_len; k++) {
			float b = grad_in[k * num_out + i];
			float *in_k = inputs + k * num_in;
			for (int j = 0; j < num_in; j++) {
				gi[j] += b * in_k[j];
			}
			gi[num_in] += b;
		}
	}

	return;
}

void forward (Network *net, float *input, float *output) {
	Layer *layer;
	layer = net->layers[0];
	int a = layer->num_in * layer->seq_len;
	memcpy(layer->outputs, input, a * sizeof(float));
	for (int i = 1; i < net->num_layers; i++) {
		layer = net->layers[i];
		layer->forward(layer);
	}
	layer = net->layers[net->num_layers - 1];
	a = layer->seq_len * layer->num_out;
	memcpy(output, layer->outputs, a * sizeof(float));
	return;
}

void backward (Network *net, float *error) {
	Layer *layer;
	for (int i = 0; i < net->num_layers - 1; i++) {
		layer = net->layers[i];
		memset(layer->grad_out, 0, sizeof(float) * layer->num_out * layer->seq_len);
	}
	layer = net->layers[net->num_layers - 1];
	int a = layer->num_out * layer->seq_len;
	for (int i = 0; i < a; i++) {
		layer->grad_out[i] = error[i];
	}
	layer->backward(layer);
	for (int i = net->num_layers - 2; i > 0; i--) {
		layer = net->layers[i];
		layer->backward(layer);
	}
	return;
}

void actualizar (Network *net, float lr, int batch_size, float lambda) {
	float beta1_t = net->beta1_t, beta2_t = net->beta2_t;
	for (int i = 0; i < net->num_layers; i++) {
		Layer *layer = net->layers[i];
		int total_w = layer->total_w;
		if (total_w == 0) continue;
		float *actual_w = layer->w;
		float *actual_m = layer->m;
		float *actual_v = layer->v;
		float *actual_grad = layer->grad;
		float learn_rate = (layer->type == 9) ? lr * 0.6f : lr;

		#pragma omp parallel for schedule(static)
		for (int j = 0; j < total_w; j++) {
			float g = actual_grad[j] / batch_size + lambda * actual_w[j];

			actual_m[j] = BETA1 * actual_m[j] + BETA1_MINUS * g;
			actual_v[j] = BETA2 * actual_v[j] + BETA2_MINUS * g * g;
			
			float m_hat = actual_m[j] / (1.0f - beta1_t);
			float v_hat = actual_v[j] / (1.0f - beta2_t);

			actual_w[j] -= learn_rate * m_hat / (sqrtf(v_hat) + EPSILON);
			actual_grad[j] = 0.0f;
		}
	}
	net->beta1_t *= BETA1;
	net->beta2_t *= BETA2;
	return;
}

void iniciar_w (Layer *layer) {
	float *a = layer->w;
	for (int i = 0; i < layer->total_w; i++) {
		*a++ = normal_random() * 0.02f;
	}
}

void iniciar_embeddings (Layer *layer) {
	float *a = layer->w;
	float scale = sqrtf(1.0f / layer->num_out);
	for (int i = 0; i < layer->total_w; i++) {
		*a++ = ((float)rand() / RAND_MAX * 2 - 1) * scale;
	}
}

int str_w_file (Network *net, char *filename) {
	FILE *f = fopen(filename, "wb");
	fwrite(&net->beta1_t, sizeof(float), 1, f);
	fwrite(&net->beta2_t, sizeof(float), 1, f);
	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = net->layers[i];
		int size = l->total_w;
		if (size > 0) {
			float *w = l->w;
			float *v = l->v;
			float *m = l->m;
			fwrite(w, sizeof(float), size, f);
			fwrite(v, sizeof(float), size, f);
			fwrite(m, sizeof(float), size, f);
		}
	}
	fclose(f);
	return 1;
}

int load_w_file (Network *net, char *filename) {
	FILE *f = fopen(filename, "rb");
	if (!f) return 0;
	int a = fread(&net->beta1_t, sizeof(float), 1, f);
	a = fread(&net->beta2_t, sizeof(float), 1, f);
	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = net->layers[i];
		int size = l->total_w;
		if (size > 0) {
			float *w = l->w;
			float *v = l->v;
			float *m = l->m;
			a = fread(w, sizeof(float), size, f);
			a = fread(v, sizeof(float), size, f);
			a = fread(m, sizeof(float), size, f);
		}
	}
	fclose(f);
	return 1;
}

int *translate (char *input, char **vocab, Merge *merges, int num_merges, int max_token_len, int vocab_size, int *out_len) {
	int len = strlen(input);
	int *tokens = malloc(sizeof(int) * len);
	int n = 0;	

	for (int i = 0; i < len; i++) {
        	char s[2] = {input[i], '\0'};
        	int found = -1;
        	for (int j = 0; j < vocab_size; j++) {
            		if (strcmp(vocab[j], s) == 0) { found = j; break; }
        	}
        	tokens[n++] = (found == -1) ? 0 : found;
    	}

    	for (int m = 0; m < num_merges; m++) {
        	int a = merges[m].a;
        	int b = merges[m].b;
        	int r = merges[m].result;
        	int new_n = 0;
        	for (int i = 0; i < n; ) {
            		if (i + 1 < n && tokens[i] == a && tokens[i + 1] == b) {
                		tokens[new_n++] = r;
                		i += 2;
            		} else {
                		tokens[new_n++] = tokens[i++];
            		}
        	}
        	n = new_n;
    	}

	int *out = malloc(sizeof(int) * n);
	memcpy(out, tokens, n * sizeof(int));
	free(tokens);
	*out_len = n;
	return out;
}

Network *defnn(int num_layers, Layer **layers) {
	Network *net = malloc(sizeof(Network));
	net->num_layers = num_layers;
	net->layers = malloc(sizeof(Layer*) * num_layers);
	for (int i = 0; i < num_layers; i++) {
        	net->layers[i] = layers[i];
	}
	net->beta1_t = BETA1;
	net->beta2_t = BETA2;
	net->forward = forward;
	net->backward = backward;
	net->actualizar = actualizar;
	return net;
}

Layer *defL_FC (int num_in, int num_out, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 0;
	layer->num_in = num_in;
	layer->num_out = num_out;
	layer->total_w = num_out * (num_in + 1);
	layer->training = 1;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;

	layer->inputs = malloc(sizeof(float*));
	layer->grad_in = malloc(sizeof(float*));
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;

	layer->grad = calloc(layer->total_w, sizeof(float));
	layer->w = malloc(sizeof(float) * layer->total_w);
	layer->v = calloc(layer->total_w, sizeof(float));
	layer->m = calloc(layer->total_w, sizeof(float));
	
	layer->outputs = malloc(sizeof(float) * num_out * seq_len);
	layer->grad_out = calloc(num_out * seq_len, sizeof(float));

	layer->x_hat = NULL;
	layer->inv_std = 0;

	layer->forward = FCforward;
	layer->backward = FCback;
	return layer;
}

Layer *defL_relu (int num_in, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 1;
	layer->num_in = num_in;
	layer->num_out = num_in;
	layer->total_w = 0;
	layer->training = 1;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;
	
	layer->inputs = malloc(sizeof(float*));
	layer->grad_in = malloc(sizeof(float*));
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;
	
	layer->w = layer->v = layer->m = layer->grad = NULL;

	layer->grad_out = calloc(num_in * seq_len, sizeof(float));
	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	
	layer->x_hat = NULL;
	layer->inv_std = 0;
	
	layer->forward = relu_forward;
        layer->backward = relu_back;
	return layer;
}

Layer *defL_sigmoid (int num_in, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 2;
	layer->num_in = num_in;
	layer->num_out = num_in;
	layer->total_w = 0;
	layer->training = 1;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;

	layer->inputs = malloc(sizeof(float*));
	layer->grad_in = malloc(sizeof(float*));
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;

	layer->w = layer->v = layer->m = layer->grad = NULL;
	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	layer->grad_out = calloc(num_in * seq_len, sizeof(float));
	
	layer->x_hat = NULL;
	layer->inv_std = 0;
	
	layer->forward = sigmoid_forward;
        layer->backward = sigmoid_back;
	return layer;
}

Layer *defL_softmax (int num_in, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 3;
	layer->num_in = num_in;
	layer->num_out = num_in;
	layer->total_w = 0;
	layer->training = 1;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;

	layer->inputs = malloc(sizeof(float*));
	layer->grad_in = malloc(sizeof(float*));
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;

	layer->w = layer->v = layer->m = layer->grad = NULL;
	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	layer->grad_out = calloc(num_in * seq_len, sizeof(float));

	layer->x_hat = NULL;
	layer->inv_std = 0;

	layer->forward = softmax_forward;
        layer->backward = softmax_back;
	return layer;
}

Layer *defL_norm (int num_in, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 6;
	layer->num_in = num_in;
	layer->num_out = num_in;
	int total_w = 2 * num_in;
	layer->total_w = total_w;
	layer->training = 1;
	layer->seq_len = seq_len;

	layer->inputs = malloc(sizeof(float*));
	layer->grad_in = malloc(sizeof(float*));
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;

	layer->grad = calloc(total_w, sizeof(float));
	layer->w = malloc(sizeof(float) * total_w);
	layer->v = calloc(total_w, sizeof(float));
	layer->m = calloc(total_w, sizeof(float));
	for (int i = 0; i < total_w; i += 2) {
		layer->w[i] = 1.0f;
		layer->w[i+1] = 0.0f;
	}
	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	layer->grad_out = calloc(num_in * seq_len, sizeof(float));

	layer->x_hat = malloc(sizeof(float) * num_in * seq_len);
	layer->inv_var = malloc(sizeof(float) * seq_len);
	layer->inv_std = 0;

	layer->forward = norm_forward;
        layer->backward = norm_back;
	return layer;
}

Layer *defL_add (int num_out, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 5;
	layer->num_in = num_out;
	layer->num_out = num_out;
	layer->total_w = 0;
	layer->training = 1;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;

	layer->w = layer->v = layer->m = layer->grad = NULL;

	layer->inputs = malloc(sizeof(float*) * 2);
	layer->inputs[0] = layers[0]->outputs;
	layer->inputs[1] = layers[1]->outputs;
	layer->grad_in = malloc(sizeof(float*) * 2);
	layer->grad_in[0] = layers[0]->grad_out;
	layer->grad_in[1] = layers[1]->grad_out;

	layer->outputs = malloc(sizeof(float) * num_out * seq_len);
	layer->grad_out = calloc(num_out * seq_len, sizeof(float));

	layer->x_hat = NULL;
	layer->inv_std = 0;

	layer->forward = add_forward;
        layer->backward = add_back;
	return layer;
}

Layer *defL_input (int num_in, int seq_len) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 4;
	layer->num_in = num_in;
	layer->num_out = num_in;
	layer->total_w = 0;
	layer->training = 1;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;

	layer->w = layer->v = layer->m = layer->grad = NULL;
	layer->inputs = layer->grad_in = NULL;

	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	layer->grad_out = calloc(num_in * seq_len, sizeof(float));
	
	layer->x_hat = NULL;
	layer->inv_std = 0;

	layer->forward = NULL;
        layer->backward = NULL;
	return layer;
}

Layer *defL_dropout(int num_in, float p, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 7;
	layer->training = 1;
	layer->num_in  = num_in;
	layer->num_out = num_in;
	layer->total_w = 0;
	layer->seq_len = seq_len;
    	layer->inv_var = NULL;

	layer->inv_std = 1.0f / (1.0f - p); //reutilizada como probabilidad inversa

	layer->inputs = malloc(sizeof(float*));
	layer->grad_in = malloc(sizeof(float*));
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;

	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	layer->grad_out = calloc(num_in * seq_len, sizeof(float));
	layer->x_hat = malloc(sizeof(float) * num_in * seq_len); //reutilizada como mask

	layer->w = layer->v = layer->m = layer->grad = NULL;

	layer->forward  = dropout_forward;
	layer->backward = dropout_back;
	return layer;
}

Layer *defL_attention (int num_in, int seq_len, int heads, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
	layer->type = 8;
	layer->training = 1;
	layer->num_in  = num_in;
	layer->num_out = num_in;
	layer->total_w = 0;
    	layer->inv_var = NULL;

	layer->seq_len = seq_len;
	layer->inv_std = heads; //reutilzada

	layer->inputs = malloc(sizeof(float*) * 3);
	layer->grad_in = malloc(sizeof(float*) * 3);
	layer->inputs[0] = layers[0]->outputs;
	layer->grad_in[0] = layers[0]->grad_out;
	layer->inputs[1] = layers[1]->outputs;
	layer->grad_in[1] = layers[1]->grad_out;
	layer->inputs[2] = layers[2]->outputs;
	layer->grad_in[2] = layers[2]->grad_out;

	layer->outputs = malloc(sizeof(float) * num_in * seq_len);
	layer->grad_out = calloc(num_in * seq_len, sizeof(float));
	layer->x_hat = malloc(heads * seq_len * seq_len * sizeof(float)); //reutilizada 

	layer->w = malloc(seq_len * seq_len * sizeof(float)); //reutilizada
	layer->m = malloc(seq_len * seq_len * sizeof(float));  //reutilizada
	layer->v = malloc(seq_len * num_in * sizeof(float));   //reutilizada
	layer->grad = NULL;

	layer->forward  = attention_forward;
	layer->backward = attention_back;
	return layer;
	
}

Layer *defL_embedding(int vocab_size, int dim, int seq_len, Layer **layers) {
	Layer *layer = malloc(sizeof(Layer));
    	layer->type = 9;
    	layer->num_in = 1;
    	layer->num_out = dim;
    	layer->total_w = dim * vocab_size;
    	layer->training = 1;
    	layer->seq_len = seq_len;
    	layer->inv_std = 0;
    	layer->x_hat = NULL;
    	layer->inv_var = malloc(sizeof(float) * seq_len * dim);

    	layer->inputs = malloc(sizeof(float*));
    	layer->grad_in = malloc(sizeof(float*));
    	layer->inputs[0] = layers[0]->outputs;
    	layer->grad_in[0] = layers[0]->grad_out;

    	layer->w = malloc(sizeof(float) * vocab_size * dim);
	layer->v = calloc(vocab_size * dim, sizeof(float));
	layer->m = calloc(vocab_size * dim, sizeof(float)); 
	layer->grad = malloc(sizeof(float) * vocab_size * dim);
    	layer->outputs = malloc(sizeof(float) * dim * seq_len);
    	layer->grad_out = calloc(dim * seq_len, sizeof(float));

	float *a = layer->inv_var;
	for (int i = 0; i < seq_len; i++) {
        	for (int j = 0; j < dim; j++) {
            		float angle = i / powf(10000.0f, (2 * (j / 2)) / (float)dim);
            		if (j % 2 == 0) {
                		*a++ = sinf(angle);
			} else {
                		*a++ = cosf(angle);
			}
        	}
    	}

    	layer->forward = embedding_forward;
    	layer->backward = embedding_back;
    	return layer;
}

void net_set_training(Network *net, int training) {
	for (int i = 0; i < net->num_layers; i++) {
        	net->layers[i]->training = training;
	}
}

void net_set_seq_len(Network *net, int seq_len) {
	for (int i = 0; i < net->num_layers; i++) {
        	net->layers[i]->seq_len = seq_len;
	}
}

void freeL(Layer *layer) {
        free(layer->grad);
        free(layer->w);
        free(layer->v);
        free(layer->m);
        free(layer->outputs);
	free(layer->grad_out);
	free(layer->inv_var);
	free(layer->inputs);
	free(layer->grad_in);
        layer->grad = NULL;
        layer->w = NULL;
        layer->m = NULL;
        layer->v = NULL;
        layer->inputs = NULL;
        layer->outputs = NULL;
	layer->grad_out = NULL;
	free(layer->x_hat);
        free(layer);
	return;
}

void freeN(Network *net) {
	for (int i = 0; i < net->num_layers; i++) {
        	freeL(net->layers[i]);
	}
	free(net->layers);
	net->layers = NULL;
	net->forward = NULL;
	net->backward = NULL;
	net->actualizar = NULL;
	free(net);
	return;
}
