from word_survival import readparagraphs, createtokens


survivalrounds = 50

textsource = '/home/sfo/store/gutenberg/gutenbooks/'

tokenoutput = '/home/sfo/data/models/tokens/text-chunks.jsonl'
#tokenoutput = False

#tokeninput = '/home/sfo/data/models/tokens/text-chunks.jsonl'

coveragetest = True
#coveragetest = False


#=== pipeline ===

paragraphs = readparagraphs(textsource)

if tokenoutput:
    tokensfile = createtokens(paragraphs, tokenoutput, survivalrounds, coveragetest)
    print('tokens written to', tokensfile)
else:
    tokensfile = tokeninput
    print('using existing tokens at', tokensfile)
