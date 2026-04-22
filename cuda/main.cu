#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// epochs actuales: 21

#define SEQ_LEN_MAX 192
#define VOCAB_SIZE 10000

#define MIN_LR 0.00001f
#define MAX_LR 0.0005f
#define WARMUP 4
#define LAMBDA 0.001f
#define TOTAL_EPOCHS 100

#define MODEL_DIM 256
#define HEADS 4
#define BATCH_SIZE 32
#define LAYERS 4
#define MAX_TOKEN_LEN 32
#define DROPOUT 0.1f
#define TEMPERATURA 0.7f

int *read_text_ints(const char *filename, int *len) {
	FILE *f = fopen(filename, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f); rewind(f);
	int *tokens = (int*)malloc(fsize);
	int a = fread(tokens, sizeof(int), fsize / sizeof(int), f);
	fclose(f);
	*len = (int)(fsize / sizeof(int));
	return tokens;
}

float lr_schedule(int step, int total, int warmup, float lr_max, float lr_min) {
	if (step < warmup) return lr_max * ((float)step / (float)warmup);
	float t = (float)(step - warmup) / (float)(total - warmup);
	if (t > 1.0f) t = 1.0f;
	return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(3.1415926f * t));
}

void shuffle_text(int *lista, int len) {
	int *tokens = (int*)malloc(sizeof(int) * (SEQ_LEN_MAX + 1));
	int a = len / (SEQ_LEN_MAX + 1);
	if (a <= 1) { free(tokens); return; }
	int stride = sizeof(int) * (SEQ_LEN_MAX + 1);
	int offset = 0;
	for (int i = 0; i < a; i++) {
		int r = i + (int)((uint32_t)xorshift64() % (unsigned)(a - i));
		r *= SEQ_LEN_MAX + 1;
		memcpy(tokens, lista + offset, stride);
		memcpy(lista + offset, lista + r, stride);
		memcpy(lista + r, tokens, stride);
		offset += SEQ_LEN_MAX + 1;
	}
	free(tokens);
}

void training(Network *net, int *h_text, int len, int epochs, int actual_epoch, const char *w_file) {
	int a = len / ((SEQ_LEN_MAX + 1) * BATCH_SIZE);
	net_set_seq_len(net, SEQ_LEN_MAX);

	int total_steps = TOTAL_EPOCHS * a;
	int step = actual_epoch * a;
	int warmup_steps = WARMUP * a;

	float *d_ftoks;
	float *d_buffer;
	int   *d_targets;
	float *d_loss;

	cudaMalloc((void**)&d_ftoks,   SEQ_LEN_MAX * sizeof(float));
	cudaMalloc((void**)&d_buffer,  SEQ_LEN_MAX * VOCAB_SIZE * sizeof(float));
	cudaMalloc((void**)&d_targets, SEQ_LEN_MAX * sizeof(int));
	cudaMalloc((void**)&d_loss,    sizeof(float));

	float *h_ftoks = (float*)malloc(SEQ_LEN_MAX * sizeof(float));
	shuffle_text(h_text, len);

	for (int i = 0; i < epochs; i++) {
		int *actual_tok = h_text;
		for (int j = 0; j < a; j++) {
			float epoch_loss = 0.0f;
			for (int k = 0; k < BATCH_SIZE; k++) {
				for (int h = 0; h < SEQ_LEN_MAX; h++)
					h_ftoks[h] = (float)actual_tok[h];
				cudaMemcpy(d_ftoks, h_ftoks, SEQ_LEN_MAX * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_targets, actual_tok + 1, SEQ_LEN_MAX * sizeof(int), cudaMemcpyHostToDevice);
				net->forward(net, d_ftoks, d_buffer);
				compute_loss_grad(d_buffer, d_targets, d_loss, SEQ_LEN_MAX, VOCAB_SIZE);
				float batch_loss;
				cudaMemcpy(&batch_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
				epoch_loss += batch_loss;
				net->backward(net, d_buffer);
				actual_tok += SEQ_LEN_MAX + 1;
			}

			float lr = lr_schedule(step, total_steps, warmup_steps, MAX_LR, MIN_LR);
			net->actualizar(net, lr, BATCH_SIZE, LAMBDA);
			step++;

			if (j % 50 == 0)
				printf("epoch %d batch %d/%d | loss: %.4f\n",
					   i, j, a, epoch_loss / (BATCH_SIZE * SEQ_LEN_MAX));
		}
		shuffle_text(h_text, len);
	}

	free(h_ftoks);
	cudaFree(d_ftoks); cudaFree(d_buffer);
	cudaFree(d_targets); cudaFree(d_loss);
}

int main(int argc, char **argv) {
	const char *w_file		= (argc > 1) ? argv[1] : "files/pesos.bin";
	const char *text_file	= (argc > 2) ? argv[2] : "files/token_text.bin";

	int num_layers = LAYERS * 14 + 4;
	Layer **layers = (Layer**)malloc(sizeof(Layer*) * num_layers);
	Layer **inputs = (Layer**)malloc(sizeof(Layer*) * 3);

	layers[0] = defL_input(1, SEQ_LEN_MAX);
	inputs[0] = layers[0];
	layers[1] = defL_embedding(VOCAB_SIZE, MODEL_DIM, SEQ_LEN_MAX, inputs);

	for (int i = 0; i < LAYERS; i++) {
		inputs[0] = layers[1 + 14 * i];
		layers[2 + i*14] = defL_norm(MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[2 + 14*i];
		layers[3 + i*14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);
		layers[4 + i*14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);
		layers[5 + i*14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[3 + i*14];
		inputs[1] = layers[4 + i*14];
		inputs[2] = layers[5 + i*14];
		layers[6 + i*14] = defL_attention(MODEL_DIM, SEQ_LEN_MAX, HEADS, inputs);

		inputs[0] = layers[6 + i*14];
		layers[7 + i*14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[7 + i*14];
		layers[8 + i*14] = defL_dropout(MODEL_DIM, DROPOUT, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[8 + 14*i];
		inputs[1] = layers[1 + 14*i];
		layers[9 + i*14] = defL_add(MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[9 + i*14];
		layers[10 + i*14] = defL_norm(MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[10 + i*14];
		layers[11 + i*14] = defL_FC(MODEL_DIM, MODEL_DIM*4, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[11 + i*14];
		layers[12 + i*14] = defL_relu(MODEL_DIM*4, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[12 + i*14];
		layers[13 + i*14] = defL_FC(MODEL_DIM*4, MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[13 + i*14];
		layers[14 + i*14] = defL_dropout(MODEL_DIM, DROPOUT, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[14 + i*14];
		inputs[1] = layers[9  + i*14];
		layers[15 + i*14] = defL_add(MODEL_DIM, SEQ_LEN_MAX, inputs);
	}
	int a = num_layers - 2;
	inputs[0] = layers[a - 1];
	layers[a++] = defL_norm(MODEL_DIM, SEQ_LEN_MAX, inputs);
	inputs[0] = layers[a - 1];
	layers[a++] = defL_FC(MODEL_DIM, VOCAB_SIZE, SEQ_LEN_MAX, inputs);

	Network *net = defnn(num_layers, layers);
	free(layers); 
	free(inputs);

	int len;
	if (!load_w_file(net, w_file)) {
		for (int i = 0; i < num_layers; i++) {
			Layer *l = net->layers[i];
			if (l->total_w > 0) iniciar_w(l);
		}
	}
	int *h_text = read_text_ints(text_file, &len);

	int epochs, actual_epoch;
	printf("introduce cantidad de epochs a entrenar: ");
	int b = scanf("%d", &epochs);
	printf("introduce en que epoch te quedaste: ");
	a = scanf("%d", &actual_epoch);
	printf("\n");

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	training(net, h_text, len, epochs, actual_epoch, w_file);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double tiempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	str_w_file(net, w_file);

	free(h_text);
	printf("entrenamiento finalizado en %.1fs (%.1fs/epoch)\n", tiempo, tiempo/epochs);
	freeN(net);
	return 0;
}
