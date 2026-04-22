CC = gcc
CFLAGS = -Iinclude -O3 -ffast-math -march=native -funroll-loops -fopenmp
LIBS = -lm -fopenmp

OBJ_MAIN = build/main.o build/nn.o
OBJ_TRANS = build/translator.o build/nn.o
OBJ_TEST = build/test.o build/nn.o

all: build/nn build/translator build/bpe_token build/test

build/nn: $(OBJ_MAIN)
	$(CC) $(OBJ_MAIN) $(LIBS) -o build/nn

build/translator: $(OBJ_TRANS)
	$(CC) $(OBJ_TRANS) $(LIBS) -o build/translator

build/test: $(OBJ_TEST)
	$(CC) $(OBJ_TEST) $(LIBS) -o build/test

build/bpe_token: src/bpe_token.c
	$(CC) -O3 -ffast-math -march=native src/bpe_token.c -lm -o build/bpe_token

build/main.o: src/main.c
	$(CC) $(CFLAGS) -c src/main.c -o build/main.o

build/nn.o: src/nn.c
	$(CC) $(CFLAGS) -c src/nn.c -o build/nn.o

build/translator.o: src/translate.c
	$(CC) $(CFLAGS) -c src/translate.c -o build/translator.o

build/test.o: src/test.c
	$(CC) $(CFLAGS) -c src/test.c -o build/test.o

run: all
	@printf "¿Qué modelo usar? [1] DIM128  [2] DIM256 : "; \
	read choice; \
	case $$choice in \
		1) MODEL="models/DIM128" ; DATA="data/processed/DIM128" ;; \
		2) MODEL="models/DIM256" ; DATA="data/processed/DIM256" ;; \
		*) echo "Opción inválida"; exit 1 ;; \
	esac; \
	mkdir -p $$MODEL $$DATA; \
	echo ">>> usando $$MODEL"; \
	if [ ! -f $$MODEL/vocab.bin ]; then \
		echo ">>> vocab.bin no encontrado, ejecutando bpe_token..."; \
		./build/bpe_token $$DATA/clean_books.txt $$MODEL/vocab.bin $$MODEL/merges.bin; \
	else \
		echo ">>> vocab.bin ya existe, saltando bpe_token"; \
	fi; \
	if [ ! -f $$DATA/token_books.bin ]; then \
		echo ">>> token_books.bin no encontrado, ejecutando translator..."; \
		./build/translator $$DATA/clean_books.txt $$DATA/token_books.bin $$MODEL/vocab.bin $$MODEL/merges.bin; \
	else \
		echo ">>> token_books.bin ya existe, saltando translator"; \
	fi; \
	echo ">>> iniciando entrenamiento..."; \
	./build/nn $$MODEL/pesos.bin $$DATA/token_books.bin $$MODEL/vocab.bin $$MODEL/merges.bin

test: build/test
	@printf "¿Qué modelo usar? [1] DIM128  [2] DIM256 : "; \
	read choice; \
	case $$choice in \
		1) DIR="models/DIM128" ;; \
		2) DIR="models/DIM256" ;; \
		*) echo "Opción inválida"; exit 1 ;; \
	esac; \
	./build/test $$DIR/pesos.bin $$DIR/vocab.bin $$DIR/merges.bin

clean:
	rm -f build/*.o build/nn build/translator build/bpe_token build/test

cleandata:
	@printf "¿Qué carpeta limpiar? [1] DIM128  [2] DIM256  [3] ambas : "; \
	read choice; \
	case $$choice in \
		1) rm -f models/DIM128/vocab.bin models/DIM128/merges.bin models/DIM128/pesos.bin data/processed/DIM128/token_books.bin ;; \
		2) rm -f models/DIM256/vocab.bin models/DIM256/merges.bin models/DIM256/pesos.bin data/processed/DIM256/token_books.bin ;; \
		3) rm -f models/DIM128/vocab.bin models/DIM128/merges.bin models/DIM128/pesos.bin data/processed/DIM128/token_books.bin \
		         models/DIM256/vocab.bin models/DIM256/merges.bin models/DIM256/pesos.bin data/processed/DIM256/token_books.bin ;; \
		*) echo "Opción inválida"; exit 1 ;; \
	esac; \
	echo ">>> archivos eliminados"
