import sys
from file_reading import read_input
from token_creation import create_tokens


survival_rounds = 50

input_path = '/home/sfo/store/gutenberg/gutenbooks/'

token_output = '/home/sfo/data/models/tokens/survival-tokens.jsonl'
#token_output = False

token_word_coverage_test = True
#token_word_coverage_test = False

#print the final token count after writing
#print_token_count = True
print_token_count = False


#=== pipeline ===

paragraphs = read_input(input_path)

if token_output:
    output_file = create_tokens(
        paragraphs,
        output_file=token_output,
        survival_rounds=survival_rounds,
        token_word_coverage_test=token_word_coverage_test,
    )
    print('tokens written to', output_file)

    if print_token_count:
        with open(output_file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print('token count:', count)
