from time import time
import os
import sys
import argparse
from pathlib import Path

# create_tokens lives at the repo root and owns the rust handoff.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from token_creation import create_tokens

# i had prior to this filtered out entire files that don't conform to english
# text or the characters im using

DEFAULT_CORPUS = '/home/sfo/store/gutenberg/gutenbooks/'
ENCODINGS = ["utf-8-sig", "cp1252", "iso-8859-1"]
ENDPUNCTUATION = ['.', '!', '?']


def read_text(path):
    for encoding in ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode file: {path}")


def split_paragraphs(text):
    paragraphs = []
    paragraph = ''
    for textblock in text.split('\n'):
        if textblock:
            paragraph += textblock + ' '
        elif paragraph:
            paragraphs.append(paragraph)
            paragraph = ''
    return paragraphs


def is_book_paragraph(paragraph):
    blockers = ['gutenberg', 'http', 'www', '.org', ' ebook']
    if any(i in paragraph.lower() for i in blockers):
        # can't be an obvious ebook signature
        return False
    punctuationcount = sum(paragraph.count(i) for i in ENDPUNCTUATION)
    if punctuationcount == 0:
        # no punctuation -> probably some ebook signature
        return False
    capitals = sum(1 for i in paragraph if i.isupper())
    lowers = sum(1 for i in paragraph if i.islower())
    if capitals + lowers == 0:
        return False
    # ratio of capitals to lowercase and capitals to punctuation should make
    # sense, otherwise it's likely not book text
    caseratio = abs(capitals - lowers) / (capitals + lowers)
    sentenceratio = abs(punctuationcount - capitals) / (punctuationcount + capitals)
    return 0.7 > sentenceratio > 0.3 and caseratio > 0.7


def load_corpus(folder):
    """Read every file in folder and return the filtered book paragraphs."""
    finaltext = []
    for file in os.listdir(folder):
        text = read_text(os.path.join(folder, file))
        finaltext.extend(p for p in split_paragraphs(text) if is_book_paragraph(p))
    return finaltext


def run_tokens(output, corpus=DEFAULT_CORPUS, survival_rounds=50):
    """Read and filter the corpus, then build the survival-token word list at
    output. Returns the output Path."""
    nt = time()
    print('file reading start')
    finaltext = load_corpus(corpus)
    print('file reading end', time() - nt)

    return create_tokens(finaltext, output, survival_rounds=survival_rounds)


def main():
    parser = argparse.ArgumentParser(
        description='Build the survival-token word list from a text corpus.')
    parser.add_argument('output', type=Path,
                        help='destination jsonl word list')
    parser.add_argument('--corpus', type=Path, default=Path(DEFAULT_CORPUS),
                        help='folder of source text files')
    parser.add_argument('--survival-rounds', type=int, default=50)
    args = parser.parse_args()

    run_tokens(args.output, corpus=args.corpus,
               survival_rounds=args.survival_rounds)


if __name__ == '__main__':
    main()
