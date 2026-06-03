import sys
from file_reading import read_input
from token_creation import create_tokens


survival_rounds = 50

input_path = '/home/sfo/store/gutenberg/gutenbooks/'

token_output = '/home/sfo/data/models/tokens/text-chunks.jsonl'
#token_output = False

#token_input_file = '/home/sfo/data/models/tokens/text-chunks.jsonl'

token_word_coverage_test = True
#token_word_coverage_test = False


#=== pipeline ===

paragraphs = read_input(input_path)

if token_output:
    tokens_file = create_tokens(
        paragraphs,
        output_file=token_output,
        survival_rounds=survival_rounds,
        token_word_coverage_test=token_word_coverage_test,
    )
    print('tokens written to', tokens_file)
else:
    tokens_file = token_input_file
    print('using existing tokens at', tokens_file)
