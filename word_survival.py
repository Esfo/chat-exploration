from time import time
from pathlib import Path
import json
import string
import subprocess

from read_paragraphs import read_paragraphs

survivalproject = Path(__file__).resolve().parent / 'survival_game'

endpunctuation = ['.', '!', '?']


def word_survival(textsource, output, survivalrounds, coveragetest=True):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    spaces = list(string.whitespace)
    punctuation = list(string.punctuation) + ['—']

    wordsplits = spaces + ['--'] + punctuation
    wordsplits = sorted(set(wordsplits), key=len, reverse=True)

    config = {
        'survivalrounds': survivalrounds,
        'wordsplits': wordsplits,
        'endpunctuation': endpunctuation,
    }

    nt = time()
    print('token survival start')

    #nothing is written to disk except the final output: the config is passed inline as a json arg and the (huge) finaltext is streamed in over stdin
    proc = subprocess.Popen(
        ['cargo', 'run', '--release', '--',
         '-', json.dumps(config, ensure_ascii=False), str(output),
         '1' if coveragetest else '0'],
        cwd=survivalproject, stdin=subprocess.PIPE, text=True, encoding='utf-8',
    )
    for paragraph in read_paragraphs(textsource):
        proc.stdin.write(json.dumps(paragraph, ensure_ascii=False) + '\n')
    proc.stdin.close()
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    print('token survival end', time() - nt)

    return output
