#!/usr/bin/env python3
"""
download_gutenberg.py — descarga automáticamente los libros más populares de Gutenberg en inglés
Uso: python3 download_gutenberg.py [--books 30] [--out carpeta]
"""

import os
import time
import argparse
import urllib.request
import urllib.error
import json

# ── IDs de los libros más populares en inglés de Project Gutenberg ───────────
POPULAR_IDS = [
    1342,  # Pride and Prejudice — Austen
    11,    # Alice in Wonderland — Carroll
    1661,  # Sherlock Holmes — Doyle
    98,    # A Tale of Two Cities — Dickens
    1400,  # Great Expectations — Dickens
    46,    # A Christmas Carol — Dickens
    2701,  # Moby Dick — Melville
    84,    # Frankenstein — Shelley
    345,   # Dracula — Stoker
    74,    # The Adventures of Tom Sawyer — Twain
    76,    # Huckleberry Finn — Twain
    1080,  # A Modest Proposal — Swift
    2600,  # War and Peace — Tolstoy
    2554,  # Crime and Punishment — Dostoevsky
    1232,  # The Prince — Machiavelli
    120,   # Treasure Island — Stevenson
    219,   # Heart of Darkness — Conrad
    1184,  # The Count of Monte Cristo — Dumas
    3207,  # The Call of the Wild — London
    35,    # The Time Machine — Wells
    36,    # The War of the Worlds — Wells
    5230,  # The Island of Doctor Moreau — Wells
    768,   # Wuthering Heights — Brontë
    1260,  # Jane Eyre — Brontë
    2148,  # The Murders in the Rue Morgue — Poe
    932,   # The Mysterious Affair at Styles — Christie
    863,   # The Mystery of the Yellow Room — Leroux
    4300,  # Ulysses — Joyce
    514,   # Little Women — Alcott
    16,    # Peter Pan — Barrie
    1952,  # The Yellow Wallpaper — Gilman
    160,   # The Importance of Being Earnest — Wilde
    174,   # The Picture of Dorian Gray — Wilde
    1727,  # The Odyssey — Homer
    158,   # Emma — Jane Austen
    161,   # Sense and Sensibility — Jane Austen
    121,   # Northanger Abbey — Jane Austen
    105,   # Persuasion — Jane Austen
    730,   # Oliver Twist — Charles Dickens
    766,   # David Copperfield — Charles Dickens
    580,   # Bleak House — Charles Dickens
    786,   # Little Dorrit — Charles Dickens
    2852,  # The Hound of the Baskervilles — Arthur Conan Doyle
    244,   # A Study in Scarlet — Arthur Conan Doyle
    834,   # The Memoirs of Sherlock Holmes — Arthur Conan Doyle
    2097,  # The Sign of the Four — Arthur Conan Doyle
    1837,  # The Prince and the Pauper — Mark Twain
    43,    # Dr Jekyll and Mr Hyde — Robert Louis Stevenson
    27827, # The Scarlet Letter — Nathaniel Hawthorne
    77,    # The House of the Seven Gables — Nathaniel Hawthorne
    209,   # The Turn of the Screw — Henry James
    910,   # White Fang — Jack London
    526,   # Lord Jim — Joseph Conrad
    159,   # The Invisible Man — H. G. Wells
    103,   # Twenty Thousand Leagues Under the Sea — Jules Verne
    1937,  # Around the World in 80 Days — Jules Verne
    1268,  # Journey to the Center of the Earth — Jules Verne
    164,   # From the Earth to the Moon — Jules Verne
    12,    # Through the Looking-Glass — Lewis Carroll
    55,    # The Wonderful Wizard of Oz — L. Frank Baum
    113,   # The Secret Garden — Frances Hodgson Burnett
    146,   # Little Lord Fauntleroy — Frances Hodgson Burnett
    45,    # Anne of Green Gables — L. M. Montgomery
    289,   # The Wind in the Willows — Kenneth Grahame
    19033, # The Tale of Peter Rabbit — Beatrix Potter
    1065,  # Poems — Edgar Allan Poe
    215,   # The Call of Cthulhu — H. P. Lovecraft
    376,   # The Dunwich Horror — H. P. Lovecraft
    10897, # Carmilla — Sheridan Le Fanu
    236,   # The Jungle Book — Rudyard Kipling
    1936,  # Just So Stories — Rudyard Kipling
    537,   # The Lost World — Arthur Conan Doyle
    325,   # Phantastes — George MacDonald
    778,   # Five Children and It — E. Nesbit
    4217,  # A Portrait of the Artist as a Young Man — James Joyce
    2814,  # Dubliners — James Joyce
    125,   # The Moonstone — Wilkie Collins
    583,   # The Woman in White — Wilkie Collins
    979,   # The Last of the Mohicans — James Fenimore Cooper
    1155,  # The Age of Innocence — Edith Wharton
    408,   # The Souls of Black Folk — W. E. B. Du Bois
    10007, # The Prisoner of Zenda — Anthony Hope
    805,   # This Side of Paradise — F. Scott Fitzgerald
    5200,  # Metamorphosis — Franz Kafka
    1998,  # Thus Spake Zarathustra — Nietzsche
    3070,  # Three Men in a Boat — Jerome K. Jerome
    848,   # The Black Arrow — Robert Louis Stevenson
    1524,  # Hamlet — Shakespeare
    28054, # The Secret Adversary — Agatha Christie
    4085,  # The Adventures of Sherlock Holmes
    500,   # The Story of Doctor Dolittle — Hugh Lofting
    829,   # Gulliver's Travels — Jonathan Swift
    205,   # Walden — Henry David Thoreau
    521,   # Robinson Crusoe — Daniel Defoe
    600,   # Notes from Underground — Dostoevsky
    708,   # Robinson Crusoe (otra edición) — Daniel Defoe
    767,   # Agnes Grey — Anne Brontë
    883,   # Our Mutual Friend — Charles Dickens
    918,   # The Fall of the House of Usher — Poe
    969,   # The Tenant of Wildfell Hall — Anne Brontë
    1040,  # The Castle of Otranto — Horace Walpole
    1131,  # The Aeneid — Virgil
    1212,  # Love and Friendship — Jane Austen
    1250,  # The Idiot — Dostoevsky
    1399,  # Anna Karenina — Tolstoy
    145,   # Middlemarch — George Eliot
    148,   # The House Behind the Cedars — Chesnutt
    157,   # Daddy-Long-Legs — Jean Webster
    135,    # Les Misérables — Victor Hugo
    2726,   # Madame Bovary — Flaubert
    22381,  # My Ántonia — Willa Cather
    10002,  # The House of Mirth — Edith Wharton
    17489,  # Ethan Frome — Edith Wharton
    8117,   # Fathers and Sons — Turgenev
    4763,   # Dead Souls — Gogol
    36034,  # The Gambler — Dostoevsky
    40745,  # Poor Folk — Dostoevsky
    28044,  # Hadji Murat — Tolstoy
    925,    # Kim — Kipling
    2226,   # Captains Courageous — Kipling
    2014,   # The Coral Island — Ballantyne
    2166,   # King Solomon's Mines — Haggard
    62,     # A Princess of Mars — Burroughs
    78,     # Tarzan of the Apes — Burroughs
    849,    # The King in Yellow — Chambers
    1647,   # The Night Land — Hodgson
    222,    # The Moon Pool — A. Merritt
    980,    # The Red House Mystery — A. A. Milne
    140,    # The Jungle — Upton Sinclair
    7300,   # The Red Badge of Courage — Stephen Crane
    7662,   # The Woodlanders — Thomas Hardy
    142,    # The Mayor of Casterbridge — Thomas Hardy
    110,    # Tess of the d’Urbervilles — Thomas Hardy
    1228,   # The Return of the Native — Thomas Hardy
    224,    # Far from the Madding Crowd — Thomas Hardy
    145,    # Middlemarch — George Eliot
    550,    # Silas Marner — George Eliot
    6688,   # The Mill on the Floss — George Eliot
    599,    # Vanity Fair — Thackeray
    2610,   # Tom Jones — Henry Fielding
    131,    # Tristram Shandy — Laurence Sterne
    25344,  # Olaudah Equiano Narrative
    3296,   # Confessions — Saint Augustine
    5740,   # The Ego and His Own — Max Stirner
    4363,   # Beyond Good and Evil — Nietzsche
    2874,   # The Book of Tea — Okakura Kakuzō
    2445,   # The Voyage Out — Virginia Woolf
    2775,   # The Good Soldier — Ford Madox Ford
    2815,   # Howards End — E. M. Forster
    2500,   # Siddhartha — Hermann Hesse
    19942,  # The Awakening — Kate Chopin
    22381,  # My Ántonia — Willa Cather
    10002,  # The House of Mirth — Edith Wharton
    17489,  # Ethan Frome — Edith Wharton
    8492,   # The Enchanted April — Elizabeth von Arnim
    403,    # The Scarlet Pimpernel — Orczy
    3825,   # Pygmalion — George Bernard Shaw
    14640,  # The Man Who Was Thursday — Chesterton
    7838,   # The Ball and the Cross — Chesterton
    3828,   # The Napoleon of Notting Hill — Chesterton
    4081,   # The Four Just Men — Edgar Wallace
    9800,   # The Angel of Terror — Edgar Wallace
    13720,  # The Clue of the Twisted Candle — Edgar Wallace
    688,    # The Haunted Hotel — Wilkie Collins
    1627,   # The Dead Secret — Wilkie Collins
    171,    # The Haunted Man — Dickens
    170,    # The Old Curiosity Shop — Dickens
    8834,   # Barnaby Rudge — Dickens
    8835,   # Sketches by Boz — Dickens
    141,    # The Warden — Anthony Trollope
    5831,   # Barchester Towers — Trollope
    421,    # Doctor Thorne — Trollope
    4121,   # Framley Parsonage — Trollope
    4731,   # The Last Chronicle of Barset — Trollope
    447,    # The Small House at Allington — Trollope
    619,    # Phineas Finn — Trollope
    3358,   # Phineas Redux — Trollope
    3628,   # The Prime Minister — Trollope
    789,    # The Eustace Diamonds — Trollope
    4230,   # An Eye for an Eye — Trollope
    864,    # Roderick Random — Tobias Smollett
    1585,   # Humphry Clinker — Smollett
    2397,   # The Captain's Daughter — Pushkin
]

MIRRORS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://gutenberg.pglaf.org/files/{id}/{id}-0.txt",
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; gutenberg-downloader/1.0)'
}

def download_book(book_id: int, out_dir: str) -> bool:
    out_path = os.path.join(out_dir, f"{book_id}.txt")
    if os.path.exists(out_path):
        print(f"  [{book_id}] ya existe, saltando")
        return True

    for url_template in MIRRORS:
        url = url_template.format(id=book_id)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
            with open(out_path, 'wb') as f:
                f.write(content)
            size_kb = len(content) / 1024
            print(f"  [{book_id}] OK — {size_kb:.0f} KB")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # probar siguiente mirror/formato
            print(f"  [{book_id}] HTTP {e.code} en {url}")
        except Exception as e:
            print(f"  [{book_id}] Error: {e}")
            continue

    print(f"  [{book_id}] FALLIDO — no se pudo descargar")
    return False


def main():
    parser = argparse.ArgumentParser(description='Descarga libros de Project Gutenberg')
    parser.add_argument('--books', type=int, default=183,
                        help='Número de libros a descargar (default: 183)')
    parser.add_argument('--out', default='gutenberg',
                        help='Carpeta de destino (default: gutenberg/)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Segundos entre descargas para no saturar el servidor (default: 2)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ids = POPULAR_IDS[:args.books]
    print(f"Descargando {len(ids)} libros en '{args.out}/'")
    print(f"Delay entre descargas: {args.delay}s")
    print()

    ok = 0
    fail = 0
    for i, book_id in enumerate(ids):
        success = download_book(book_id, args.out)
        if success:
            ok += 1
        else:
            fail += 1
        if i < len(ids) - 1:
            time.sleep(args.delay)

    print()
    print(f"Descargados : {ok}")
    print(f"Fallidos    : {fail}")

if __name__ == '__main__':
    main()
