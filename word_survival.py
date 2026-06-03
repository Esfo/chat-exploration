import os
import json
import string
import subprocess
from time import time
from pathlib import Path

encodings = ["utf-8-sig", "cp1252", "iso-8859-1"]
endpunctuation = ['.', '!', '?']
blockers = ['gutenberg', 'http', 'www', '.org', ' ebook']

survivalproject = Path(__file__).resolve().parent / 'survival_game'


def readparagraphs(textsource):
    source = Path(textsource)
    files = [source] if source.is_file() else [source / name for name in os.listdir(source)]
    for filepath in files:
        text = None
        for encoding in encodings:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise UnicodeError(f"could not decode file: {filepath}")

        paragraph = ''
        for textblock in (text + '\n').split('\n'):
            if textblock:
                paragraph += textblock + ' '
            elif paragraph:
                lowered = paragraph.lower()
                punctuationcount = sum(paragraph.count(p) for p in endpunctuation)
                capitals = sum(1 for ch in paragraph if ch.isupper())
                lowers = sum(1 for ch in paragraph if ch.islower())
                letters = capitals + lowers
                if (not any(b in lowered for b in blockers)
                        and punctuationcount and letters
                        and abs(capitals - lowers) / letters > 0.7
                        and 0.7 > abs(punctuationcount - capitals) / (punctuationcount + capitals) > 0.3):
                    yield paragraph
                paragraph = ''


def createtokens(paragraphs, output, survivalrounds, coveragetest=True):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    wordsplits = list(string.whitespace) + ['--'] + list(string.punctuation) + ['—']
    wordsplits = sorted(set(wordsplits), key=len, reverse=True)

    config = {
        'survivalrounds': survivalrounds,
        'wordsplits': wordsplits,
        'endpunctuation': endpunctuation,
    }

    nt = time()
    print('token survival start')

    proc = subprocess.Popen(
        ['cargo', 'run', '--release', '--',
         '-', json.dumps(config, ensure_ascii=False), str(output),
         '1' if coveragetest else '0'],
        cwd=survivalproject, stdin=subprocess.PIPE, text=True, encoding='utf-8',
    )
    for paragraph in paragraphs:
        proc.stdin.write(json.dumps(paragraph, ensure_ascii=False) + '\n')
    proc.stdin.close()
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    print('token survival end', time() - nt)
    return output
