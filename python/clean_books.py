#!/usr/bin/env python3
"""
clean_books.py — concatena y limpia libros de Project Gutenberg
Uso: python3 clean_books.py libro1.txt libro2.txt ... -o output.txt [--max-mb 10]
     python3 clean_books.py carpeta/ -o output.txt [--max-mb 10]
"""

import sys
import re
import os
import argparse
import unicodedata

# ── Regex compilados ─────────────────────────────────────────────────────────
RE_URL       = re.compile(r'https?://\S+|www\.\S+')
RE_EMAIL     = re.compile(r'\S+@\S+\.\S+')
RE_NUMBER    = re.compile(r'\b\d+(\.\d+)?\b')
RE_BRACKET   = re.compile(r'\[.*?\]')               # referencias [1], [Nota]
RE_PUNCT_BAD = re.compile(r'[^a-z .,?!\'\-@$\n ]') # todo lo que no queremos
RE_SPACES    = re.compile(r'[ \t]+')
RE_MULTIDOL  = re.compile(r'(\$\s*){2,}')           # $ repetidos

MIN_LINE_WORDS = 5

# ── Cabeceras y pies de Gutenberg ────────────────────────────────────────────
GUTENBERG_START = [
    "*** start of",
    "*** end of",
    "*end*",
    "project gutenberg",
    "gutenberg license",
    "gutenberg-tm",
    "produced by",
    "transcribed by",
    "this ebook",
    "end of the project",
    "end of project",
    "proofreading team",
]

def is_gutenberg_meta(line: str) -> bool:
    l = line.lower()
    return any(marker in l for marker in GUTENBERG_START)

# ── Normalización ASCII ───────────────────────────────────────────────────────
def to_ascii(s: str) -> str:
    """Convierte tildes y caracteres unicode a su equivalente ASCII más cercano."""
    normalized = unicodedata.normalize('NFKD', s)
    return normalized.encode('ascii', 'ignore').decode('ascii')

# ── Limpieza de un fragmento de texto ────────────────────────────────────────
def clean_text(text: str) -> str:
    # Saltar metadatos de Gutenberg
    if is_gutenberg_meta(text):
        return ""

    # Quitar URLs y emails
    text = RE_URL.sub(' ', text)
    text = RE_EMAIL.sub(' ', text)

    # Quitar referencias entre corchetes [1], [nota]
    text = RE_BRACKET.sub(' ', text)

    # Normalizar unicode → ASCII
    text = to_ascii(text)

    # Lowercase
    text = text.lower()

    # Números → @
    text = RE_NUMBER.sub('@', text)

    # Quitar todo carácter no permitido
    # Permitidos: a-z espacio . , ? ! ' - @ $
    text = RE_PUNCT_BAD.sub(' ', text)

    # Colapsar espacios
    text = RE_SPACES.sub(' ', text).strip()

    # Descartar párrafos demasiado cortos
    if len(text.split()) < MIN_LINE_WORDS:
        return ""

    return text

# ── Procesar un archivo ───────────────────────────────────────────────────────
def process_file(path: str, out, max_bytes: int, written: int) -> int:
    print(f"  Procesando: {os.path.basename(path)}")
    lines_out = 0
    paragraph = []

    def emit(para_lines):
        # Une todas las líneas del párrafo en una sola
        return ' '.join(para_lines)

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            stripped = raw.strip()

            if stripped == '':
                # Línea vacía → fin de párrafo
                if paragraph:
                    full = emit(paragraph)
                    paragraph = []
                    cleaned = clean_text(full)
                    if not cleaned:
                        continue
                    token_line = cleaned + ' $\n'
                    encoded = token_line.encode('ascii', 'ignore')
                    if written + len(encoded) > max_bytes:
                        return written, True
                    out.write(token_line)
                    written += len(encoded)
                    lines_out += 1
            else:
                paragraph.append(stripped)

        # Emitir último párrafo si no termina en línea vacía
        if paragraph:
            full = emit(paragraph)
            cleaned = clean_text(full)
            if cleaned:
                token_line = cleaned + ' $\n'
                encoded = token_line.encode('ascii', 'ignore')
                if written + len(encoded) <= max_bytes:
                    out.write(token_line)
                    written += len(encoded)
                    lines_out += 1

    print(f"    → {lines_out:,} párrafos | total acumulado: {written / 1e6:.2f} MB")
    return written, False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Limpia y concatena libros de Gutenberg')
    parser.add_argument('inputs', nargs='+', help='.txt individuales o carpeta con .txt')
    parser.add_argument('-o', '--output', default='clean_books.txt', help='Archivo de salida')
    parser.add_argument('--max-mb', type=float, default=10.0, help='MB máximos de output (default: 10)')
    parser.add_argument('--min-words', type=int, default=5, help='Palabras mínimas por línea (default: 5)')
    args = parser.parse_args()

    global MIN_LINE_WORDS
    MIN_LINE_WORDS = args.min_words
    max_bytes = int(args.max_mb * 1024 * 1024)

    # Recopilar archivos
    files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            for fname in sorted(os.listdir(inp)):
                if fname.endswith('.txt'):
                    files.append(os.path.join(inp, fname))
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            print(f"Advertencia: '{inp}' no existe, ignorando")

    if not files:
        print("Error: no se encontraron archivos .txt")
        sys.exit(1)

    print(f"Archivos encontrados : {len(files)}")
    print(f"Output               : {args.output}")
    print(f"Límite               : {args.max_mb} MB")
    print()

    written = 0
    done = False

    with open(args.output, 'w', encoding='ascii') as out:
        for fpath in files:
            written, done = process_file(fpath, out, max_bytes, written)
            if done:
                print(f"\nLímite de {args.max_mb} MB alcanzado.")
                break

    print()
    print(f"Tamaño final : {written / 1e6:.2f} MB")
    print(f"Guardado en  : {args.output}")

if __name__ == '__main__':
    main()
