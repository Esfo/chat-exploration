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


#hierarchical survival game  (text-search / tree-descent variant)
#
#same game as hierarchical-survival.py, but the per-round occurrence counting is done
#completely differently for speed.
#
#the slow version enumerated EVERY substring of every paragraph up to some length - that
#is O(textlen * maxlen) of pure-python slicing and hashing per paragraph, most of it spent
#on garbage substrings that no living token cares about.
#
#instead we walk the LIVING TREE top-down against the text and only ever look where a
#living branch actually reaches:
#  - scan the text once for every length-2 substring and remember the positions it occurs
#    at (this is the only full pass over the text).
#  - then, level by level, take each token we found and - only if it was alive coming into
#    the round - look at the single character to its left and right at each of its
#    occurrences. those are its length+1 extensions. a parent that does NOT appear is never
#    descended into, so whole dead subtrees cost nothing.
#  - the extensions we find are exactly this branch's children: the ones that are already
#    living tokens get their occurrence count, and any extension we have never seen before
#    is a brand-new child to be born on top of this parent. (this is the "if the leaves
#    don't account for all of the parent's occurrences, make the missing ones" idea.)
#
#everything downstream - births, the leaf-only +1/-1 survival game, and the final
#consolidation - is identical to hierarchical-survival.py.

deathfloor = 0   #a leaf dies when its survival score drops to this

nt = time()
print('hierarchical survival game begin')

#we only split into words to build `allwords` for the coverage test; the game itself runs
#on the raw paragraph text so spaces/punctuation are real, linkable characters.
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

for text in finaltext:
    #words are only needed for the coverage sanity check, not for the game itself
    allwords.update(part for part in re.split(wordpattern, text) if part)

    textlen = len(text)

    #base layer - count every single character (spaces included) in one pass
    singles.update(text)

    #snapshot of who was alive coming INTO this round. only these branches get descended
    #into, so a token (re)born this round cannot grow a child this round - branches climb
    #exactly one level per round, off a parent that already proved it was alive.
    prevalive = frozenset(score)

    #--- descend the living tree against the text ---------------------------------------
    #present[token] = number of times this token occurs in this text. populated only for
    #tokens that are either a length-2 substring or an extension of a living branch.
    present = {}

    #current level: token -> set of start positions where it occurs in the text.
    #seed with every length-2 substring (the one and only full scan of the text).
    current = defaultdict(set)
    for i in range(textlen - 1):
        current[text[i:i + 2]].add(i)

    level = 2
    while current:
        nxt = defaultdict(set)
        for token, positions in current.items():
            present[token] = len(positions)

            #only living branches grow; a parent that wasn't alive coming in is left as a
            #tip for this round (its own birth/scoring still happens below)
            if token not in prevalive:
                continue

            #the children of this branch are its one-character left/right extensions, read
            #straight off the parent's occurrence positions. using a set of start positions
            #dedupes an extension that is reachable from both its prefix and suffix parent.
            for i in positions:
                if i + level < textlen:
                    nxt[text[i:i + level + 1]].add(i)        #extend right
                if i - 1 >= 0:
                    nxt[text[i - 1:i + level]].add(i - 1)    #extend left

        current = nxt
        level += 1

    #--- births + raw counts -----------------------------------------------------------
    for token, occ in present.items():
        if token in score:
            #already living - just accrue its raw occurrences (score handled below)
            count[token] += occ
            continue

        #a brand-new token. length-2 tokens grow straight off the permanent base layer; any
        #longer token only ended up in `present` because it extended a living (prevalive)
        #parent, so it is guaranteed eligible - we just recover which parent(s) to link it to.
        if len(token) == 2:
            livingparents = ()
        else:
            livingparents = [p for p in (token[:-1], token[1:]) if p in prevalive]
            if not livingparents:
                continue   #safety; shouldn't happen given how `present` is built

        #born as a fresh leaf; the survival game below gives it its first +1
        score[token] = deathfloor
        count[token] = occ
        leaves.add(token)
        parents[token] = set(livingparents)
        for p in livingparents:
            children[p].add(token)
            leaves.discard(p)   #parent now has a child -> frozen, off the frontier

    #--- the survival game - only the frontier leaves play it --------------------------
    for token in tuple(leaves):
        if token in present:
            score[token] += 1
        else:
            score[token] -= 1
            if score[token] <= deathfloor:
                #this branch tip is dead - remove it and detach from its parents,
                #re-promoting any parent that has now lost its last child
                leaves.discard(token)
                del score[token]
                del count[token]
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
