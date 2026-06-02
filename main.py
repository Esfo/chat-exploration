import sys
from file_reading import read_input
from token_creation import create_tokens

# === config (numbers) ===
survival_rounds = 50

# === switches: comment out the line you don't want; the active line wins ===

# token output location: a string path writes the jsonl there, False skips it
token_output = '/home/sfo/data/models/tokens/'
# token_output = False

# word-completion (coverage) test: prints how many words the tokens can rebuild
run_coverage_test = True
# run_coverage_test = False

# clear the output directory before writing
# clear_output = True
clear_output = False

# print the final token count after writing
# print_token_count = True
print_token_count = False


# === pipeline ===
if len(sys.argv) < 2:
    print('usage: python main.py <file-or-folder>')
    sys.exit(1)

input_path = sys.argv[1]

if clear_output and token_output:
    from pathlib import Path
    import shutil
    out = Path(token_output)
    if out.exists():
        shutil.rmtree(out)

paragraphs = read_input(input_path)

if token_output:
    output_file = create_tokens(
        paragraphs,
        output_path=token_output,
        survival_rounds=survival_rounds,
        run_coverage_test=run_coverage_test,
    )
    print('tokens written to', output_file)

    if print_token_count:
        with open(output_file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print('token count:', count)
