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

    nt = time()
    print('token survival start')

    # Nothing is written to disk except the final output: the config is passed
    # inline as a JSON arg and the (huge) finaltext is streamed in over stdin.
    proc = subprocess.Popen(
        ['cargo', 'run', '--release', '--',
         '-', json.dumps(config, ensure_ascii=False), str(output_file),
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
