from read_paragraphs import read_paragraphs
from word_survival import word_survival


survivalrounds = 50

textsource = '/home/sfo/store/gutenberg/gutenbooks/'

tokenoutput = '/home/sfo/data/models/tokens/text-chunks.jsonl'
#tokenoutput = False

#tokeninput = '/home/sfo/data/models/tokens/text-chunks.jsonl'

coveragetest = True
#coveragetest = False


#=== pipeline ===

paragraphs = read_paragraphs(textsource)

if tokenoutput:
    tokensfile = word_survival(paragraphs, tokenoutput, survivalrounds, coveragetest)
    print('tokens written to', tokensfile)
else:
    tokensfile = tokeninput
    print('using existing tokens at', tokensfile)
