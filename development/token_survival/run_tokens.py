from collections import Counter, defaultdict
from time import time
import itertools
import string
import os
import re
import json
import subprocess
from pathlib import Path

#i had prior to this filtered out entire files that don't conform to english text or the characters im using

folder = '/home/sfo/store/gutenberg/gutenbooks/'
outputfolder = '/home/sfo/data/models/tokens/'

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
                #ratio of capitals to lowercase and capitals to punctuation should probably make sense, otherwise it's likely not book text
                caseratio = abs(capitals-lowers)/(capitals+lowers)
                sentenceratio = abs(punctuationcount-capitals)/(punctuationcount+capitals)
                if 0.7 > sentenceratio > 0.3 and caseratio > 0.7:
                    finaltext.append(paragraph)

print('token drafting end', time() - nt)

nt = time()
print('rust handoff start')

project = Path(__file__).resolve().parent
datafolder = project / 'data'
outputfolder = project / 'output'

datafolder.mkdir(exist_ok=True)
outputfolder.mkdir(exist_ok=True)

wordsplits = spaces + ['--'] + punctuation
wordsplits = sorted(set(wordsplits), key=len, reverse=True)

config = {
    'survivalrounds': survivalrounds,
    'wordsplits': wordsplits,
    'endpunctuation': endpunctuation,
}

finaltextpath = datafolder / 'finaltext.jsonl'
configpath = datafolder / 'config.json'

with open(finaltextpath, 'w', encoding='utf-8') as f:
    for paragraph in finaltext:
        f.write(json.dumps(paragraph, ensure_ascii=False) + '\n')

with open(configpath, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False)

subprocess.run(
    [
        'cargo',
        'run',
        '--release',
        '--',
        str(finaltextpath),
        str(configpath),
        str(outputfolder),
    ],
    cwd=project,
    check=True,
)

print('rust handoff end', time() - nt)
