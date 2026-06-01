from collections import Counter, defaultdict
from time import time
import itertools
import string
import os
import re


def find_index(word, target):
    start = 0
    indices = []
    tlen = len(target)
    while True:
        i = word.find(target, start)
        if i == -1:
            break
        indices.append([i,i+tlen])
        start = i + 1
    return indices

def word_processing(folder):
    
    files = os.listdir(folder)

    encodings = ["utf-8-sig", "cp1252", "iso-8859-1"]

    digits = list(string.digits)
    spaces = list(string.whitespace) #replace these prior i guess
    punctuation = list(string.punctuation) + ['—']
    newline = ['\n']

    pretokens = digits + spaces + punctuation + newline
    pattern = '(' + '|'.join(re.escape(char) for char in pretokens) + ')'


    nt = time()
    print('file reading start')

    wordcounts = Counter()
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
            raise UnicodeError(f"Could not decode file: {filename}")

        text = re.sub(r"\n+", lambda m: " " if len(m.group(0)) == 1 else m.group(0), text)
        wordcounts += Counter(part for part in re.split(pattern, text) if part)

    print('file reading end', time() - nt)

    return wordcounts


def token_drafting(wordcounts):
    
    nt = time()
    print('token drafting start')

    wordsbytoken = defaultdict(set) #token: [words]
    combinationcounts = Counter()
    for word, count in wordcounts.items():
        if len(word) > 1:
            #every combination of adjacent letters of a word broken into pieces
            for size in range(2, len(word) + 1):
                for start in range(0, len(word) - size + 1):
                    token = word[start:start + size]
                    combinationcounts[token] += count
                    wordsbytoken[token].add(word)
        else:
            combinationcounts[word] += count
            wordsbytoken[word].add(word)

    print('token drafting end', time() - nt)

    nt = time()
    print('token drafting cleanup start')

    for k in wordsbytoken:
        wordsbytoken[k] = tuple(wordsbytoken[k])

    wordsbytoken = dict(wordsbytoken)
    wordcounts = list(wordcounts)

    tokensbyidentifier = {} #tokenid: token
    identifiersbytoken = {} #token: tokenid
    for n, (combination, counts) in enumerate(combinationcounts.most_common(len(combinationcounts))):
        tokensbyidentifier[n] = combination
        identifiersbytoken[combination] = n

    print('end token drafting cleanup', time() - nt)

    return combinationcounts, wordsbytoken, wordcounts, tokensbyidentifier, identifiersbytoken


def token_fitting(combinationcounts, wordsbytoken, wordcounts, tokensbyidentifier, identifiersbytoken):
    
    nt = time()
    print('token fitting start')

    wordedges = defaultdict(lambda: defaultdict(list)) #word: startindex: [(endindex, tokenid), ...]

    coveredwords = set() #words with a completed token path

    coveragegoal = len(wordcounts) #stopping point
    allowedtokens = [] #final token set

    for token, tokenid in identifiersbytoken.items():
        allowedtokens.append(token)

        changedwords = set() #words whose edge map changed because this token was just added

        #mapping potential tokens to their appropriate starting points
        #Katherine: 2=t: [(endindex=4, th) (endindex=5, the)
        for word in wordsbytoken[token]:
            if word in coveredwords:
                continue
            for start, end in find_index(word, token):
                edge = (end, tokenid)

                if edge not in wordedges[word][start]:
                    wordedges[word][start].append(edge)
                    changedwords.add(word)

        #rebuild coverage for words affected by the newly added token
        for word in changedwords:
            wlen = len(word)
            stack = [(0, [])]
            foundcoverage = False

            #walk forward through token edges; a valid path must start at 0 and end at len(word)
            while stack and not foundcoverage:
                position, tokenpath = stack.pop()

                if position == wlen:
                    foundcoverage = True
                    continue

                for end, nexttokenid in wordedges[word].get(position, []):
                    stack.append((end, tokenpath + [nexttokenid]))

            if foundcoverage:
                coveredwords.add(word)

        if len(coveredwords) == coveragegoal:
            break

    print('token fitting end', time() - nt)

    return tuple(allowedtokens)

def create_tokens(folder):
    return token_fitting(*token_drafting(word_processing(folder)))
