from collections import Counter, defaultdict
from time import time
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
print('file reading start')

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

print('file reading end', time() - nt)


#hierarchical survival game
#
#instead of every subword fighting independently (which lets a short subword survive
#purely because it rides along inside many unrelated supersets), we grow a tree.
#
#every unique character is a permanent "base layer" - it can never be removed, not even
#a space. the multi-character tokens are grown on top of these as a hierarchy:
#  a length-2 token (e.g. "ie") is a child of its single characters
#  a length-3 token (e.g. "tie" or "iet") is grown by adding one character to either
#    end of a surviving length-2 token
#  a length-4 token is grown the same way from a surviving length-3 token, and so on
#    with no ceiling, as long as the extended branch keeps surviving
#
#each token plays a cumulative survival game:
#  - while it appears in a text it gains its occurrence count
#  - while it is absent it loses 1 per round, and after `survivalrounds` consecutive
#    misses (with a non-positive count) it dies
#  - BUT a parent is protected: it never takes the -1 penalty while it still has at
#    least one living child branch. it can only start decaying once every child is gone.
#
#a token can only be born once at least one of its (length-1) end-substrings is already
#a living token, so branches genuinely build 2 -> 3 -> 4 -> ... upward.

nt = time()
print('hierarchical survival game begin')

wordsplits = spaces + ['--'] + punctuation
wordsplits = sorted(set(wordsplits), key=len, reverse=True)
pattern = '(' + '|'.join(map(re.escape, wordsplits)) + ')'

allwords = set()
singles = Counter()              #the permanent base layer: char -> count (never removed)
survivors = Counter()            #living multi-character tokens: token -> count
misses = Counter()               #token -> consecutive rounds missed
children = defaultdict(set)      #token -> set of living child tokens grown from it
parents = {}                     #token -> set of the (len-1) end-substrings it grew from
tokencache = {}                  #word -> Counter of every substring it contains

for text in finaltext:
    #separating out words
    words = [part for part in re.split(pattern, text) if part]
    wordcounts = Counter(words)
    allwords.update(wordcounts)

    #count every substring present in this text (all lengths at once)
    modifications = Counter()
    for word, wordcount in wordcounts.items():
        cached = tokencache.get(word)
        if cached is None:
            cached = Counter()
            for size in range(1, len(word) + 1):
                for start in range(0, len(word) - size + 1):
                    cached[word[start:start + size]] += 1
            tokencache[word] = cached
        for token, tokencount in cached.items():
            modifications[token] += tokencount * wordcount

    #process tokens shortest-first so a parent born this round is available to its
    #children in the very same round (this is what lets branches cascade upward)
    for token in sorted(modifications, key=len):
        count = modifications[token]

        if len(token) == 1:
            #base layer - always exists, simply accrues its count
            singles[token] += count
            continue

        if token in survivors:
            #already living - feed it and reset its miss streak
            survivors[token] += count
            misses.pop(token, None)
            continue

        #candidate for birth: it needs a living parent end-substring.
        #length-2 tokens grow straight off the permanent base layer, so they always qualify.
        prefix = token[:-1]   #drop last char
        suffix = token[1:]    #drop first char
        livingparents = set()
        if len(token) == 2:
            #parents are single characters (the base layer) - implicitly alive
            livingparents.update(p for p in (prefix, suffix) if p in singles or len(p) == 1)
        else:
            if prefix in survivors:
                livingparents.add(prefix)
            if suffix in survivors:
                livingparents.add(suffix)

        if not livingparents:
            #no living branch to grow from yet - it stays unborn for now
            continue

        #born
        survivors[token] = count
        misses.pop(token, None)
        parents[token] = set(livingparents)
        for p in livingparents:
            if len(p) > 1:
                children[p].add(token)

    #killing the unworthy (with parent protection)
    for token in tuple(survivors):
        if token in modifications:
            continue

        #protected: a parent never decays while it still has a living child branch
        if children.get(token):
            continue

        if survivors[token] > 0:
            survivors[token] -= 1
        else:
            misses[token] += 1
            if misses[token] >= survivalrounds:
                #this branch tip is dead - detach it from its parents so they can
                #eventually start decaying once all their own children are gone
                for p in parents.pop(token, ()):
                    childset = children.get(p)
                    if childset is not None:
                        childset.discard(token)
                        if not childset:
                            del children[p]
                children.pop(token, None)
                del survivors[token]
                del misses[token]

print('hierarchical survival game end', time() - nt)


#consolidation
#
#if a parent's count is identical to one of its children's count, then that parent only
#ever occurs as part of that child - the parent carries no information of its own, so it
#is deleted and the longer child is kept moving forward.
#e.g. if count("die") == count("diet") then only "diet" survives.
#the base layer (single characters) is never deleted.

nt = time()
print('consolidation start')

removed = set()
#read all counts before mutating so chains (di -> die -> diet) resolve consistently
for parent in tuple(survivors):
    for child in children.get(parent, ()):
        if child in survivors and survivors[parent] == survivors[child]:
            removed.add(parent)
            break

for parent in removed:
    del survivors[parent]

print('consolidated away:', len(removed))
print('consolidation end', time() - nt)


#final token set = permanent base layer + surviving grown tokens
nt = time()
print('token coverage test start')

allwords = tuple(allwords)
tokens = tuple(set(singles) | set(survivors))

print('base layer (single chars):', len(singles))
print('surviving multi tokens:', len(survivors))
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

#since every single character is a permanent base-layer token, every word is guaranteed
#to be coverable - the coverage test here is really just a sanity check.
