#ifndef NN_H
#define NN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <stdint.h>

typedef struct Layer {
	int type;
	int num_in;
	int num_out;
	int total_w;
	int training;

	int seq_len;

	float *x_hat;
	float inv_std;
	float *inv_var;

	float *w;
	float *v;
	float *m;
	float *grad;

	float **inputs;
	float **grad_in;
	float *outputs;
	float *grad_out;

	void (*forward) (struct Layer*);
	void (*backward) (struct Layer*);
} Layer;

typedef struct Network {
	int num_layers;
	Layer **layers;
	float beta1_t;
	float beta2_t;

	void (*forward) (struct Network*, float *input, float *output);
	void (*backward) (struct Network*, float *error);
	void (*actualizar) (struct Network*, float lr, int batch_size, float lambda);
} Network;

typedef struct {
    	int a;
    	int b;
    	int result;
} Merge;

Network *defnn(int num_layers, Layer **layers);
Layer *defL_FC(int num_in, int num_out, int seq_len, Layer **layers);
Layer *defL_relu(int num_in, int seq_len, Layer **layers);
Layer *defL_sigmoid(int num_in, int seq_len, Layer **layers);
Layer *defL_softmax(int num_in, int seq_len, Layer **layers);
Layer *defL_add(int num_out, int seq_len, Layer **layers);
Layer *defL_norm(int num_in, int seq_len, Layer **layers);
Layer *defL_dropout(int num_in, float p, int seq_len, Layer **layers);
Layer *defL_input(int num_in, int seq_len);
Layer *defL_attention(int num_in, int seq_len, int heads, Layer **layers);
Layer *defL_embedding(int vocab_size, int dim, int seq_len, Layer **layers);

void net_set_training(Network *net, int training);
void net_set_seq_len(Network *net, int seq_len);

void iniciar_w (Layer *layer); 
void iniciar_embeddings(Layer *layer); 
int str_w_file (Network *net, const char *filename);
int load_w_file (Network *net, const char *filename);
void compute_loss_grad(float *d_logits, int *d_targets, float *d_loss, int seq_len, int vocab_size);

int *translate (char *s, char **vocab, Merge *merges, int num_merges, int max_token_len, int vocab_size, int *out_len);
uint64_t xorshift64(void);
void softmax(float *in, float *out, int len);

void freeL (Layer *layer);
void freeN (Network *net);

#endif

#ifdef __cplusplus
}
#endif
