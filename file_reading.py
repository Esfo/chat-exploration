from time import time
from pathlib import Path
import os

ENCODINGS = ["utf-8-sig", "cp1252", "iso-8859-1"]
ENDPUNCTUATION = ['.', '!', '?']
BLOCKERS = ['gutenberg', 'http', 'www', '.org', ' ebook']


def _read_text(path):
    for encoding in ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode file: {path}")


def _extract_paragraphs(text):
    paragraphs = []
    paragraph = ''
    for textblock in text.split('\n'):
        if textblock:
            paragraph += textblock + ' '
        elif paragraph:
            paragraphs.append(paragraph)
            paragraph = ''
    if paragraph:
        paragraphs.append(paragraph)
    return paragraphs


def _passes_filters(paragraph):
    if any(b in paragraph.lower() for b in BLOCKERS):
        return False
    punctuationcount = sum(paragraph.count(p) for p in ENDPUNCTUATION)
    if punctuationcount == 0:
        return False
    capitals = sum(1 for ch in paragraph if ch.isupper())
    lowers = sum(1 for ch in paragraph if ch.islower())
    if capitals + lowers == 0:
        return False
    caseratio = abs(capitals - lowers) / (capitals + lowers)
    sentenceratio = abs(punctuationcount - capitals) / (punctuationcount + capitals)
    return 0.7 > sentenceratio > 0.3 and caseratio > 0.7


def read_input(path):
    """Read a single file or every file in a folder; return filtered paragraphs.
    The argument can be a file path or a directory path — it's auto-detected."""
    nt = time()
    print('file reading start')

    p = Path(path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = [p / name for name in os.listdir(p)]
    else:
        raise FileNotFoundError(f"not a file or directory: {path}")

    paragraphs = []
    for filepath in files:
        text = _read_text(filepath)
        for paragraph in _extract_paragraphs(text):
            if _passes_filters(paragraph):
                paragraphs.append(paragraph)

    print('file reading end', time() - nt)
    return paragraphs
