#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define TARGET_VOCAB 10000
#define MAX_TOKEN_LEN 32

static char vocab[TARGET_VOCAB][MAX_TOKEN_LEN];
static int vocab_size = 0;

typedef struct {
    	int a;
    	int b;
    	int result;
} Merge;

Merge *merges = NULL;
int num_merges = 0;

static int is_valid(int id) {
	if (id < 0 || id >= vocab_size) return 0;
    	char *s = vocab[id];
    	if (s[0] == '$') return 0;
    	if (s[0] == '@') return 0;
    	return 1;
}

static int vocab_add (char *s) {
	if (strlen(s) == 0) return -1;
	for (int i = 0; i < vocab_size; i++) {
		if (strcmp(vocab[i], s) == 0) return i;
	}
	if (vocab_size >= TARGET_VOCAB) {
		fprintf(stderr, "Error: vocabulario lleno\n");
		exit(1);
	}
	strncpy(vocab[vocab_size], s, MAX_TOKEN_LEN - 1);
	vocab[vocab_size][MAX_TOKEN_LEN - 1] = '\0';
	return vocab_size++;
}

static int *corpus = NULL;
static long corp_len = 0;

int main (int argc, char *argv[]) {
	const char *infile = (argc > 1) ? argv[1] : "clean_wiki.txt";
	const char *vocab_file = (argc > 2) ? argv[2] : "vocab.bin";
	const char *merges_file = (argc > 3) ? argv[3] : "merges.bin";

	FILE *f = fopen(infile, "rb");
	if (!f) return 1;
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	rewind(f);

	char *text = malloc(fsize + 1);
	int a = fread(text, 1, fsize, f);
	fclose(f);
	text[fsize] = '\0';

	corpus = malloc(sizeof(int) * fsize);
	int char_to_id[256];
	memset(char_to_id, -1, sizeof(char_to_id));
	int id_dollar = -1;
	int id_newline = -1;

	for (long i = 0; i < fsize; i++) {
		unsigned char c = (unsigned char)text[i];
		if (char_to_id[c] == -1) {
			char s[2] = {(char)c, '\0'};
			char_to_id[c] = vocab_add(s);
			if (c == '$') id_dollar = char_to_id[c];
			if (c == '\n') id_newline = char_to_id[c];
		}
		corpus[corp_len++] = char_to_id[c];
	}
	free(text);

	printf("Vocab inicial : %d caracteres únicos\n", vocab_size);
    	printf("Corpus : %ld tokens\n", corp_len);
    	printf("Objetivo : %d tokens  (%d fusiones)\n\n", TARGET_VOCAB, TARGET_VOCAB - vocab_size);

	int merge_n = 0;

	while (vocab_size < TARGET_VOCAB) {
		int vs = vocab_size;
		long *cnt = calloc((long)vs * vs, sizeof(long));
		long best_count = 0;
		int best_a = -1, best_b = -1;
		int prev = -1;

		for (int i = 0; i < corp_len; i++) {
			int cur = corpus[i];
			if (cur == -1) continue;
			if (is_valid(prev) && is_valid(cur)) {
				if (cur == id_newline) {
    					prev = -1;
    					continue;
				}
				long idx = (long)prev * vs + cur;
				cnt[idx]++;
				if (cnt[idx] > best_count) {
					best_count = cnt[idx];
					best_a = prev;
					best_b = cur;
				}
			}
			prev = cur;
		}
		free(cnt);

		if (best_count < 2) {
			printf("no quedan pares con frecuencia mayor que 1\n");
			break;
		}

		char new_tok[MAX_TOKEN_LEN];
        	snprintf(new_tok, MAX_TOKEN_LEN, "%s%s", vocab[best_a], vocab[best_b]);
        	int new_id = vocab_add(new_tok);

        	printf("[%4d] '%s' + '%s'  ->  '%s'  (freq %ld)\n", ++merge_n, vocab[best_a], vocab[best_b], new_tok, best_count);

		merges = realloc(merges, sizeof(Merge) * (num_merges + 1));
		merges[num_merges++] = (Merge){best_a, best_b, new_id};

		long prev_idx = -1;
		int prev_id = -1;

		for (long i = 0; i < corp_len; i++) {
            		int cur = corpus[i];
            		if (cur == -1) continue;

            		if (prev_id == best_a && cur == best_b) {
                		corpus[prev_idx] = new_id;
                		corpus[i]        = -1;
                		prev_id  = -1;
                		prev_idx = -1;
            		} else {
                		prev_idx = i;
                		prev_id  = cur;
            		}
        	}

		long new_len = 0;
		for (long i = 0; i < corp_len; i++) {
    			if (corpus[i] != -1)
        			corpus[new_len++] = corpus[i];
		}
		corp_len = new_len;
	}
	FILE *fv = fopen(vocab_file, "wb");
	fwrite(&vocab_size, sizeof(int), 1, fv);
    	for (int i = 0; i < vocab_size; i++) {
    		int len = strlen(vocab[i]);
		if (len == 0) continue;
    		fwrite(&len, sizeof(int), 1, fv);
    		fwrite(vocab[i], 1, len, fv);
	}
    	fclose(fv);

	FILE *fm = fopen(merges_file, "wb");
	fwrite(&num_merges, sizeof(int), 1, fm);
	fwrite(merges, sizeof(Merge), num_merges, fm);
	fclose(fm);

    	printf("\n%d tokens guardados en '%s'\n", vocab_size, vocab_file);

    	free(corpus);
    	return 0;
}























