import sys
from file_reading import read_input
from token_creation import create_tokens

# === config (paths and numbers) ===
output_path = '/home/sfo/data/models/tokens/'
survival_rounds = 50

# === True/False switches ===
# comment out the line you don't want; the active line wins

# clear_data_dir = True
clear_data_dir = False

# print_token_count = True
print_token_count = False


# === pipeline ===
if len(sys.argv) < 2:
    print('usage: python main.py <file-or-folder>')
    sys.exit(1)

input_path = sys.argv[1]

if clear_data_dir:
    from pathlib import Path
    import shutil
    out = Path(output_path)
    if out.exists():
        shutil.rmtree(out)

paragraphs = read_input(input_path)
output_file = create_tokens(
    paragraphs,
    output_path=output_path,
    survival_rounds=survival_rounds,
)

print('tokens written to', output_file)

if print_token_count:
    with open(output_file, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    print('token count:', count)
