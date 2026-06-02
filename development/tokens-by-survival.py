from collections import Counter, defaultdict
from time import time
import itertools
import string
import os
import re

#i had prior to this filtered out entire files that don't conform to english text or the characters im using

folder = '/home/sfo/store/gutenberg/gutenbooks/'
files = os.listdir(folder)

encodings = ["utf-8-sig", "cp1252", "iso-8859-1"]

digits = list(string.digits)
spaces = list(string.whitespace) #replace these prior i guess
punctuation = list(string.punctuation) + ['—']
endpunctuation = ['.', '!', '?']
newline = ['\n']

pretokens = digits + spaces + punctuation + newline
pattern = '(' + '|'.join(re.escape(char) for char in pretokens) + ')'

survivalrounds = 50

nt = time()
print('token drafting start')

finaltext = []
for file in files:
    path = folder + file

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeError(f"Could not decode file: {file}")
    
    text = text.split('\n')
    
    paragraphs = []
    paragraph = ''
    pswitch = False
    for textblock in text:
        if textblock:
            if pswitch:
                paragraph += textblock + ' '
                continue
            else:
                paragraph += textblock + ' '
                pswitch = True
        else:
            #new paragraph
            if paragraph:
                paragraphs.append(paragraph)
            paragraph = ''
            pswitch = False
        
    blockers = ['gutenberg', 'http', 'www', '.org', ' ebook']
    for paragraph in paragraphs:
        if any(i in paragraph.lower() for i in blockers):
            #can't be an obvious ebook signature
            continue
        else:
            punctuationcount = sum(paragraph.count(i) for i in endpunctuation)
            if punctuationcount == 0:
                #if the paragraph has no punctuation it's probably some ebook signature
                continue
            capitals = sum(1 for i in paragraph if i.isupper())
            lowers = sum(1 for i in paragraph if i.islower())
            if capitals + lowers > 0:
                #ratio of capitals to lowercase and capitals to punctuation should probably make sense, othewise it's likely not book text
                caseratio = abs(capitals-lowers)/(capitals+lowers)
                sentenceratio = abs(punctuationcount-capitals)/(punctuationcount+capitals)
                if 0.7 > sentenceratio > 0.3 and caseratio > 0.7:
                    finaltext.append(paragraph)

print('token drafting end', time() - nt)


nt = time()
print('token survival game begin')

wordsplits = spaces + ['--'] + punctuation
wordsplits = sorted(set(wordsplits), key=len, reverse=True)

pattern = '(' + '|'.join(map(re.escape, wordsplits)) + ')'

allwords = set()
misses = Counter() #tokens not found in a round: number of failed rounds
survivors = Counter() #all surviving tokens tracked via count
tokencache = {} #word: cache of all contained tokens
times = Counter()
for text in finaltext:
    #separating out words
    t1 = time()
    words = [part for part in re.split(pattern, text) if part]
    wordcounts = Counter(words)
    allwords.update(wordcounts)
    modifications = Counter() #token: count in text
    times[1] += time() - t1
    
    t2 = time()
    #counting spaces
    survivors[' '] += text.count(' ')
    modifications[' '] += text.count(' ')
    
    #counting punctuation
    for p in endpunctuation:
        survivors[p] += text.count(p)
        modifications[p] += text.count(p)
    times[2] += time() - t2

    t3 = time()
    #counting token presence
    for word, wordcount in wordcounts.items():
        cached = tokencache.get(word)
        if cached is None:
            cached = Counter()
            for size in range(1, len(word) + 1):
                for start in range(0, len(word) - size + 1):
                    token = word[start:start + size]
                    cached[token] += 1
            tokencache[word] = cached
        for token, tokencount in cached.items():
            count = tokencount * wordcount
            modifications[token] += count
            survivors[token] += count
    times[3] += time() - t3

    t4 = time()
    #killing the unworthy
    for token in tuple(survivors):
        if token in modifications:
            if token in misses:
                del misses[token]
        else:
            if survivors[token] <= 0:
                misses[token] += 1
                if misses[token] >= survivalrounds:
                    del survivors[token]
                    del misses[token]
            else:
                survivors[token] -= 1
    times[4] += time() - t4

print('token survival game end', time() - nt)


nt = time()
print('middle sort start')

tokens = []
sortedtokens = survivors.most_common(len(survivors)) #sorting
for n, (s, c) in enumerate(sortedtokens):
    if n % 2:
        tokens.insert(0, s)
    else:
        tokens.append(s)

print('middle sort finished', time() - nt)


#from matplotlib import pyplot as plt
#
#lengths = [len(i) for i in tokens]
#plt.plot(list(range(len(tokens))), lengths, '.')
#plt.show()


#do survival of the fittest
#every combination, with the max len being maxwordlength-1 on that line, so that for every word
#keep them, round after round
#duplicates add more too
#
#have a dying rate of 10
#so if it shows up 1 round, +1
#if it doesnt show up in 10, -1, and this could delete it if it doesn't come back
#
#this happens through every file

#they'll all be > 2 length and just include singles as independent things after
#and singles should include capitals

#middle-sort tokens:
#highest count tokens go in the middle, and stack pyramidally outwards
#should put singles here then, perhaps spaces in the very middle?

#this is preparation for the array
#this shouldn't matter, but maybe it does
#it can be determined later if this helps or hurts the high dimensionality gradient descent, or perhaps doesnt make any difference

#record all independent words and check they can all be reconstructed via the selected token list
#lowest survival rate that achieves this is the winner i guess
#this should be independent of the single-lengths i think


nt = time()
print('token coverage test start')

allwords = tuple(allwords)
tokens = tuple(survivors)

print('total tokens:', len(tokens))
print('total words:', len(allwords))

wordedges = defaultdict(lambda: defaultdict(list)) #word: start index: [(end index, token), ...]
coveredwords = set() #words with a completed token path
uncoveredwords = set() #words without a completed token path
tokenpathsbyword = {} #word: token path that completes it

for word in allwords:
    #map every token to every place it fits in this word
    for token in tokens:
        start = word.find(token)

        while start != -1:
            end = start + len(token)
            wordedges[word][start].append((end, token))
            start = word.find(token, start + 1)

    wlen = len(word)
    stack = [(0, [])]
    seen = set()
    foundcoverage = False

    #walk forward through token edges; a valid path must start at 0 and end at len(word)
    while stack and not foundcoverage:
        position, tokenpath = stack.pop()

        if position == wlen:
            coveredwords.add(word)
            tokenpathsbyword[word] = tuple(tokenpath)
            foundcoverage = True
            continue

        if position in seen:
            continue

        seen.add(position)

        for end, nexttoken in wordedges[word].get(position, []):
            stack.append((end, tokenpath + [nexttoken]))

    if not foundcoverage:
        uncoveredwords.add(word)

print('covered words:', len(coveredwords), '/', len(allwords))
print('percent covered:', len(coveredwords) / len(allwords))
print('uncovered words:', len(uncoveredwords))
print('token coverage test end', time() - nt)

#this is cool and all, but slow, i need it to be faster, so i'll turn this into a python-rust combo
