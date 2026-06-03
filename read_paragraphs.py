from pathlib import Path
import os

#i had prior to this filtered out entire files that don't conform to english text or the characters im using

encodings = ["utf-8-sig", "cp1252", "iso-8859-1"]
endpunctuation = ['.', '!', '?']
blockers = ['gutenberg', 'http', 'www', '.org', ' ebook']


def read_paragraphs(textsource):
    source = Path(textsource)
    files = [source] if source.is_file() else [source / name for name in os.listdir(source)]

    for path in files:
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeError(f"Could not decode file: {path}")

        paragraph = ''
        for textblock in (text + '\n').split('\n'):
            if textblock:
                paragraph += textblock + ' '
                continue
            #new paragraph
            if any(i in paragraph.lower() for i in blockers):
                #can't be an obvious ebook signature
                paragraph = ''
                continue
            punctuationcount = sum(paragraph.count(i) for i in endpunctuation)
            if punctuationcount == 0:
                #if the paragraph has no punctuation it's probably some ebook signature
                paragraph = ''
                continue
            capitals = sum(1 for i in paragraph if i.isupper())
            lowers = sum(1 for i in paragraph if i.islower())
            if capitals + lowers > 0:
                #ratio of capitals to lowercase and capitals to punctuation should probably make sense, othewise it's likely not book text
                caseratio = abs(capitals-lowers)/(capitals+lowers)
                sentenceratio = abs(punctuationcount-capitals)/(punctuationcount+capitals)
                if 0.7 > sentenceratio > 0.3 and caseratio > 0.7:
                    yield paragraph
            paragraph = ''
