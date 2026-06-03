from time import time
from pathlib import Path
import json
import string
import subprocess

RUST_PROJECT = Path(__file__).resolve().parent / 'token_survival'


def create_tokens(paragraphs, output_file, survival_rounds=50, token_word_coverage_test=True):
    """Run the rust token survival binary on the given paragraphs.
    Writes the jsonl to output_file (full path, name designated by the caller)
    and returns its Path."""
    output_file = Path(output_file)
    datafolder = RUST_PROJECT / 'data'
    datafolder.mkdir(exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    spaces = list(string.whitespace)
    punctuation = list(string.punctuation) + ['—']
    endpunctuation = ['.', '!', '?']

    wordsplits = spaces + ['--'] + punctuation
    wordsplits = sorted(set(wordsplits), key=len, reverse=True)

    config = {
        'survivalrounds': survival_rounds,
        'wordsplits': wordsplits,
        'endpunctuation': endpunctuation,
    }

    configpath = datafolder / 'config.json'

    nt = time()
    print('token survival start')

    with open(configpath, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)

    # The finaltext input is huge and is only scratch input for the rust binary.
    # Stream it in over stdin ("-") so it is never written to disk at all; the
    # only output is the designated word list at output_file.
    proc = subprocess.Popen(
        ['cargo', 'run', '--release', '--',
         '-', str(configpath), str(output_file),
         '1' if token_word_coverage_test else '0'],
        cwd=RUST_PROJECT, stdin=subprocess.PIPE, text=True, encoding='utf-8',
    )
    for paragraph in paragraphs:
        proc.stdin.write(json.dumps(paragraph, ensure_ascii=False) + '\n')
    proc.stdin.close()
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    print('token survival end', time() - nt)

    return output_file
