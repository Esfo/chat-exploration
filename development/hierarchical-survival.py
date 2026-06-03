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
#the survival game only ever touches the FRONTIER - the leaf tips of each branch:
#  - a leaf that appears this round gains +1 to its survival score
#  - a leaf that is absent loses 1, and dies once its score drops to 0
#  - an internal node (a parent that still has a living child) is frozen: it gets
#    neither +1 nor -1. it only re-enters the game once every one of its children
#    has died, at which point it becomes a leaf again.
#this is the whole point of the hierarchy for performance: each round we iterate only
#the leaves, never the full survivor set.
#
#two separate numbers are tracked per token:
#  - score: the +1/-1 survival score above, which decides life and death
#  - count: the raw cumulative occurrence count, which is only used for consolidation
#
#a token can only be born once at least one of its (length-1) end-substrings was already
#a living token coming INTO the round (see prevalive below). this means branches grow at
#most one level deeper per round - a freshly (re)born length-2 cannot immediately spawn a
#length-3 in the same round; its parent has to have survived from an earlier round first.
#so branches genuinely build 2 -> 3 -> 4 -> ... upward over time.

deathfloor = 0   #a leaf dies when its survival score drops to this

nt = time()
print('hierarchical survival game begin')

#the survival game runs on the RAW paragraph text - spaces and punctuation included - so
#that " " and friends are real, linkable characters that tokens can grow across (e.g.
#"the ", " of "). we only split into words to build `allwords` for the coverage test.
wordsplits = spaces + ['--'] + punctuation
wordsplits = sorted(set(wordsplits), key=len, reverse=True)
wordpattern = '(' + '|'.join(map(re.escape, wordsplits)) + ')'

allwords = set()
singles = Counter()              #the permanent base layer: char -> count (never removed)
count = Counter()                #living multi-char token -> raw cumulative occurrences
score = {}                       #living multi-char token -> +1/-1 survival score
leaves = set()                   #the frontier: living tokens with no living children
children = defaultdict(set)      #token -> set of living child tokens grown from it
parents = {}                     #token -> set of the (len-1) end-substrings it grew from
lengthcounts = Counter()         #token length -> how many living tokens have it

for text in finaltext:
    #words are only needed for the coverage sanity check, not for the game itself
    allwords.update(part for part in re.split(wordpattern, text) if part)

    #only enumerate one level deeper than the deepest CURRENTLY living token. this tracks
    #the living set (so it shrinks again when long branches die), and thanks to the
    #one-level-per-round rule it stays small - just bigrams early on.
    maxlen = (max(lengthcounts) + 1) if lengthcounts else 2

    #enumerate every substring of the raw text up to maxlen, straight off the raw
    #character stream so spaces/punctuation are real, linkable characters.
    modifications = Counter()
    textlen = len(text)
    for size in range(1, maxlen + 1):
        for start in range(0, textlen - size + 1):
            modifications[text[start:start + size]] += 1

    #snapshot of who was alive coming INTO this round. births are gated against this
    #snapshot (not the live `score`), so a parent that is (re)born this round cannot also
    #parent a longer child this round - each branch can only grow one level deeper per
    #round, and only off a parent that already proved it was alive beforehand.
    prevalive = frozenset(score)

    #births + raw counts, shortest-first
    for token in sorted(modifications, key=len):
        occ = modifications[token]

        if len(token) == 1:
            #base layer - always exists, simply accrues its count
            singles[token] += occ
            continue

        if token in score:
            #already living - just accrue its raw occurrences (score handled below)
            count[token] += occ
            continue

        #candidate for birth: it needs a living parent end-substring.
        #length-2 tokens grow straight off the permanent base layer, so they always qualify.
        prefix = token[:-1]   #drop last char
        suffix = token[1:]    #drop first char
        if len(token) == 2:
            livingparents = ()                      #parents are base-layer singles
        else:
            livingparents = [p for p in (prefix, suffix) if p in prevalive]
            if not livingparents:
                #no parent that was already alive coming into this round - stay unborn
                continue

        #born as a fresh leaf; the survival game below gives it its first +1
        score[token] = deathfloor
        count[token] = occ
        leaves.add(token)
        parents[token] = set(livingparents)
        lengthcounts[len(token)] += 1
        for p in livingparents:
            children[p].add(token)
            leaves.discard(p)   #parent now has a child -> frozen, off the frontier

    #the survival game - only the frontier leaves play it
    for token in tuple(leaves):
        if token in modifications:
            score[token] += 1
        else:
            score[token] -= 1
            if score[token] <= deathfloor:
                #this branch tip is dead - remove it and detach from its parents,
                #re-promoting any parent that has now lost its last child
                leaves.discard(token)
                del score[token]
                del count[token]
                lengthcounts[len(token)] -= 1
                if lengthcounts[len(token)] <= 0:
                    del lengthcounts[len(token)]
                for p in parents.pop(token, ()):
                    childset = children.get(p)
                    if childset is not None:
                        childset.discard(token)
                        if not childset:
                            del children[p]
                            if p in score:           #parent still alive -> back to frontier
                                leaves.add(p)
                children.pop(token, None)

print('hierarchical survival game end', time() - nt)

#`survivors` is just the set of living multi-char tokens, keyed by raw count for the
#coverage test and consolidation below
survivors = count


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
