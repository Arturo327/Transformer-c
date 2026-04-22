#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define SEQ_LEN_MAX 192
#define VOCAB_SIZE 10000

#define MODEL_DIM 256
#define HEADS 4
#define LAYERS 4
#define MAX_TOKEN_LEN 32
#define TEMPERATURA 0.8f

int main (int argc, char **argv) {
	char *w_file = (argc > 1) ? argv[1] : "files/pesos.bin";
	char *vocab_file = (argc > 2) ? argv[2] : "files/vocab.bin";
	char *merges_file = (argc > 3) ? argv[3] : "files/merges.bin";

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
		layers[8 + i * 14] = defL_dropout(MODEL_DIM, 0.1, SEQ_LEN_MAX, inputs);

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
		layers[14 + i * 14] = defL_dropout(MODEL_DIM, 0.1, SEQ_LEN_MAX, inputs);

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
		printf("no hay pesos, entrenar primero\n");
		return 1;
	}
	net_set_training(net, 0);

	printf("# to quit\n");

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

	char *s = malloc(SEQ_LEN_MAX * MAX_TOKEN_LEN * sizeof(char));
	float *test_inputs = malloc(sizeof(float) * SEQ_LEN_MAX);
	float *test_outputs = malloc(sizeof(float) * SEQ_LEN_MAX * vocab_size);

	FILE *fa = fopen(merges_file, "rb");
	int num_merges;
	p = fread(&num_merges, sizeof(int), 1, fa);
	Merge *merges = malloc(sizeof(Merge) * num_merges);
	p = fread(merges, sizeof(Merge), num_merges, fa);
	fclose(fa);

    	fclose(f);
	srand((unsigned)time(NULL));

	float *logits = malloc(sizeof(float) * VOCAB_SIZE);
	float *probs  = malloc(sizeof(float) * VOCAB_SIZE);
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
	free(logits);
	free(probs);
	return 0;
}

















