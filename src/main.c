#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define SEQ_LEN_MAX 192
#define MODEL_DIM 256
#define LAYERS 4
#define HEADS 4
#define VOCAB_SIZE 10000

#define MIN_LR 0.00001f
#define MAX_LR 0.0005f
#define LAMBDA 0.001f
#define DROPOUT 0.1f
#define TOTAL_EPOCHS 100
#define WARMUP 4

#define BATCH_SIZE 32
#define MAX_TOKEN_LEN 32
#define TEMPERATURA 0.7f

int *read_text_ints (char *filename, int *len) {
	FILE *f = fopen(filename, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	rewind(f);
	int *tokens = malloc(fsize);
	int a = fread(tokens, sizeof(int), fsize / sizeof(int), f);
	fclose(f);
	*len = (int)(fsize / sizeof(int));
	return tokens;
}

float lr_schedule(int step, int total_steps, int warmup_steps, float lr_max, float lr_min) {
    	if (step < warmup_steps) {
        	return lr_max * ((float)step / (float)warmup_steps);
    	}
    	float t = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    	if (t > 1.0f) t = 1.0f;
    	float cosine = cosf(3.1415926f * t);
    	return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosine);
}

void shuffle_text (int *lista, int len) {
	int *tokens = malloc(sizeof(int) * (SEQ_LEN_MAX + 1));
	int a = len / (SEQ_LEN_MAX + 1);
	if (a <= 1) return;
    	int stride = sizeof(int) * (SEQ_LEN_MAX + 1);
	int offset = 0;
	for (int i = 0; i < a; i++) {
    		int random = i + (int)((uint32_t)xorshift64() % (unsigned)(a - i));
		random *= SEQ_LEN_MAX + 1;
    		memcpy(tokens, lista + offset, stride);
    		memcpy(lista + offset, lista + random, stride);
    		memcpy(lista + random, tokens, stride);
		offset += SEQ_LEN_MAX + 1;
	}	
	free(tokens);
	return;
}

void training (Network *net, int *text, int len, int epochs, int actual_epoch) {
	int *actual_tok;
	float *buffer = malloc(sizeof(float) * VOCAB_SIZE * SEQ_LEN_MAX);
	int a = len / ((SEQ_LEN_MAX + 1) * BATCH_SIZE);
	net_set_seq_len(net, SEQ_LEN_MAX);
	int total_steps = TOTAL_EPOCHS * a;
	int step = actual_epoch * a;
	int warmup = WARMUP * a;
	shuffle_text(text, len);
	for (int i = 0; i < epochs; i++) {
		actual_tok = text;
		for (int j = 0; j < a; j++) {
			float epoch_loss = 0.0f;
			for (int k = 0; k < BATCH_SIZE; k++) {
				float ftoks[SEQ_LEN_MAX];
				for (int h = 0; h < SEQ_LEN_MAX; h++) {
					ftoks[h] = (float)actual_tok[h];
				}
				net->forward(net, ftoks, buffer);

				#pragma omp parallel for reduction(+:epoch_loss) schedule(static)
				for (int h = 0; h < SEQ_LEN_MAX; h++) {
					float probs[VOCAB_SIZE];
					float *buf = buffer + VOCAB_SIZE * h;
    					int target = actual_tok[h + 1];
    					softmax(buf, probs, VOCAB_SIZE);

    					float p = probs[target];
    					epoch_loss -= logf(p + 1e-9f);

    					for (int l = 0; l < VOCAB_SIZE; l++) {
        					buf[l] = probs[l];
    					}
    					buf[target] -= 1.0f;
				}

				net->backward(net, buffer);
				actual_tok += SEQ_LEN_MAX + 1;
			}
			float lr = lr_schedule(step, total_steps, warmup, MAX_LR, MIN_LR);
			net->actualizar(net, lr, BATCH_SIZE, LAMBDA);
			step++;
			if (j % 50 == 0) {
        			printf("epoch %d batch %d/%d | loss: %.4f\n", i, j, a, epoch_loss / (BATCH_SIZE * SEQ_LEN_MAX));
			}
		}
		shuffle_text(text, len);
	}
	free(buffer);
}

int main (int argc, char **argv) {
	char *w_file = (argc > 1) ? argv[1] : "files/pesos.bin";
	char *text_file = (argc > 2) ? argv[2] : "files/token_text.bin";
	char *vocab_file = (argc > 3) ? argv[3] : "files/vocab.bin";
	char *merges_file = (argc > 4) ? argv[4] : "files/merges.bin";

	int num_layers = LAYERS * 14 + 4;
	Layer **layers = malloc(sizeof(Layer*) * num_layers);
	Layer **inputs = malloc(sizeof(Layer*) * 3);

	layers[0] = defL_input(1, SEQ_LEN_MAX);

	inputs[0] = layers[0];
	layers[1] = defL_embedding(VOCAB_SIZE, MODEL_DIM, SEQ_LEN_MAX, inputs);

	for (int i = 0; i < LAYERS; i++) {
		inputs[0] = layers[1 + 14 * i];
		layers[2 + i * 14] = defL_norm(MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[2 + 14 * i];
		layers[3 + i * 14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);
		layers[4 + i * 14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);
		layers[5 + i * 14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);
		
		inputs[0] = layers[3 + i * 14];
		inputs[1] = layers[4 + i * 14];
		inputs[2] = layers[5 + i * 14];
		layers[6 + i * 14] = defL_attention(MODEL_DIM, SEQ_LEN_MAX, HEADS, inputs);
		
		inputs[0] = layers[6 + i * 14];
		layers[7 + i * 14] = defL_FC(MODEL_DIM, MODEL_DIM, SEQ_LEN_MAX, inputs);
		
		inputs[0] = layers[7 + i * 14];
		layers[8 + i * 14] = defL_dropout(MODEL_DIM, DROPOUT, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[8 + 14 * i];
		inputs[1] = layers[1 + 14 * i];
		layers[9 + i * 14] = defL_add(MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[9 + i * 14];
		layers[10 + i * 14] = defL_norm(MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[10 + i * 14];
		layers[11 + i * 14] = defL_FC(MODEL_DIM, MODEL_DIM * 4, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[11 + i * 14];
		layers[12 + i * 14] = defL_relu(MODEL_DIM * 4, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[12 + i * 14];
		layers[13 + i * 14] = defL_FC(MODEL_DIM * 4, MODEL_DIM, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[13 + i * 14];
		layers[14 + i * 14] = defL_dropout(MODEL_DIM, DROPOUT, SEQ_LEN_MAX, inputs);

		inputs[0] = layers[14 + i * 14];
		inputs[1] = layers[9 + i * 14];
		layers[15 + i * 14] = defL_add(MODEL_DIM, SEQ_LEN_MAX, inputs);
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
			if (l->total_w > 0) {
				iniciar_w(l);
			}
		}
	}

	int *text = read_text_ints(text_file, &len);

	int epochs;
	printf("introduce cantidad de epochs a entrenar: ");
	int c = scanf("%d", &epochs);
	int actual_epoch;
	printf("introduce en que epoch te quedaste: ");
	int b = scanf("%d", &actual_epoch);
	printf("\n");

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	training (net, text, len, epochs, actual_epoch);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double tiempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

	(void)str_w_file(net, w_file);
	free(text);

	net_set_training(net, 0);

	printf("entrenamiento finalizado en %.1fs (%.1fs/epoch)\n", tiempo, tiempo / epochs);
	printf("-----TEST SPACE-----\n");
	printf("# to quit\n");
	char *s = malloc(SEQ_LEN_MAX * MAX_TOKEN_LEN * sizeof(char));
	float *test_inputs = malloc(sizeof(float) * SEQ_LEN_MAX);
	float *test_outputs = malloc(sizeof(float) * SEQ_LEN_MAX * VOCAB_SIZE);

    	FILE *f = fopen(vocab_file, "rb");
	if (!f) return 1;
	int vocab_size;
	int p = fread(&vocab_size, sizeof(int), 1, f);

	char **vocab = malloc(sizeof(char*) * vocab_size);
	for (int i = 0; i < vocab_size; i++) {
    		int len;
    		p = fread(&len, sizeof(int), 1, f);
    		vocab[i] = malloc(len + 1);
    		int p = fread(vocab[i], 1, len, f);
    		vocab[i][len] = '\0';
	}

	FILE *fa = fopen(merges_file, "rb");
	int num_merges;
	p = fread(&num_merges, sizeof(int), 1, fa);
	Merge *merges = malloc(sizeof(Merge) * num_merges);
	p = fread(merges, sizeof(Merge), num_merges, fa);
	fclose(fa);

    	fclose(f);
	srand((unsigned)time(NULL));
	while (1) {
		if (!fgets(s, SEQ_LEN_MAX * MAX_TOKEN_LEN, stdin)) break;
		s[strcspn(s, "\n")] = '\0';
		if (s[0] == '\0') continue;
		if (s[0] == '#') break;
		int *tokens = translate(s, vocab, merges, num_merges, MAX_TOKEN_LEN, vocab_size, &len);
		if (!tokens || len == 0) {
    			printf("(no se reconocieron tokens)\n");
    			free(tokens);
    			continue;
		}
		net_set_seq_len(net, len);
		for (int i = 0; i < len; i++) {
			test_inputs[i] = (float)tokens[i];
		}
		printf("\n");
		printf("%s", s);
		while (1) {
			net->forward(net, test_inputs, test_outputs);
			float temperature = TEMPERATURA;
			float *a = test_outputs + (len - 1) * vocab_size;
			
			float logits[VOCAB_SIZE];
			float probs[VOCAB_SIZE];
			for (int i = 0; i < VOCAB_SIZE; i++) {
    				logits[i] = a[i] / temperature;
			}
			softmax(logits, probs, VOCAB_SIZE);

			float r = (float)rand() / (float)RAND_MAX;
			int tok = 0;
			float cumul = 0.0f;
			for (int i = 0; i < VOCAB_SIZE; i++) {
    				cumul += probs[i];
    				if (cumul >= r) { 
					tok = i; 
					break; 
				}
			}
			if (tok < 0 || tok >= VOCAB_SIZE) break;
			if (vocab[tok][0] == '$') break;
			printf("%s", vocab[tok]);
			len++;
			net_set_seq_len(net, len);
			if (len > SEQ_LEN_MAX) break;
			test_inputs[len - 1] = (float)tok;
		}
		free(tokens);
		printf("\n\n");
	}
	for (int i = 0; i < VOCAB_SIZE; i++) {
		free(vocab[i]);
	}
	free(vocab);
	freeN(net);
	free(s);
	free(merges);
	free(test_inputs);
	free(test_outputs);
	return 0;
}

















