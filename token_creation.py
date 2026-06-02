from time import time
from pathlib import Path
import json
import string
import subprocess

RUST_PROJECT = Path(__file__).resolve().parent / 'development' / 'token_survival'


def create_tokens(paragraphs, output_path, survival_rounds=50):
    """Run the rust token survival binary on the given paragraphs.
    Writes middle_tokens.jsonl into output_path and returns its Path."""
    output_path = Path(output_path)
    datafolder = RUST_PROJECT / 'data'
    datafolder.mkdir(exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

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

    finaltextpath = datafolder / 'finaltext.jsonl'
    configpath = datafolder / 'config.json'

    nt = time()
    print('token survival start')

    with open(finaltextpath, 'w', encoding='utf-8') as f:
        for paragraph in paragraphs:
            f.write(json.dumps(paragraph, ensure_ascii=False) + '\n')

    with open(configpath, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)

    subprocess.run(
        ['cargo', 'run', '--release', '--',
         str(finaltextpath), str(configpath), str(output_path)],
        cwd=RUST_PROJECT, check=True,
    )

    print('token survival end', time() - nt)

    return output_path / 'middle_tokens.jsonl'
