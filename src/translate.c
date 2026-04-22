#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKEN_LEN 32

char *read_file(const char *filename) {
    	FILE *f = fopen(filename, "rb");
    	if (!f) {
        	perror("Error abriendo archivo");
        	return NULL;
    	}
	fseek(f, 0, SEEK_END);
    	long size = ftell(f);
    	rewind(f);

    	char *buffer = malloc(size + 1);
    	if (!buffer) {
        	fclose(f);
        	return NULL;
    	}

    	int a = fread(buffer, 1, size, f);
    	buffer[size] = '\0';

    	fclose(f);
    	return buffer;
}

int main (int argc, char *argv[]) {
	char *infile = (argc > 1) ? argv[1] : "files/clean_wiki.txt";
	char *outfile = (argc > 2) ? argv[2] : "files/token_wiki.bin";
	char *vocab_file = (argc > 3) ? argv[3] : "files/vocab.bin";
	char *merges_file = (argc > 4) ? argv[4] : "files/merges.bin";
	
	char *s = read_file(infile);
	int size = 0;

    	FILE *f = fopen(vocab_file, "rb");
	if (!f) return 1;

    	int vs = 0;
	int vocab_size;
	int a = fread(&vocab_size, sizeof(int), 1, f);
	char **vocab = malloc(sizeof(char*) * vocab_size);
    	while (vs < vocab_size) {
    		int len;
    		if (fread(&len, sizeof(int), 1, f) != 1) break;
    		vocab[vs] = malloc(len + 1);
    		int a = fread(vocab[vs], 1, len, f);
    		vocab[vs][len] = '\0';
    		vs++;
	}

    	fclose(f);

	Merge *merges = NULL;
	FILE *fa = fopen(merges_file, "rb");
	if (!fa) return 1;
	int num_merges;
	int b = fread(&num_merges, sizeof(int), 1, fa);
	merges = malloc(sizeof(Merge) * num_merges);
	b = fread(merges, sizeof(Merge), num_merges, fa);
	fclose(fa);

	int *idxs = translate(s, vocab, merges, num_merges, MAX_TOKEN_LEN, vocab_size, &size);
	free(s);
	FILE *fm = fopen(outfile, "wb");
	fwrite(idxs, sizeof(int), size, fm);
	free(idxs);
	for (int i = 0; i < vocab_size; i++) {
		free(vocab[i]);
	}
	free(vocab);
	free(merges);
	fclose(fm);
	return 0;
}








