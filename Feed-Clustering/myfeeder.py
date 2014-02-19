import feedparser
import re
from collections import defaultdict

def showEntry(e):
    print '-'*30
    print 'Title: %s' % e.title.encode('utf-8')
    print 'Author: %s' % e.author.encode('utf-8')
    if 'summary' in e:
        summary = e.summary.encode('utf-8')
    else:
        summary = e.description.encode('utf-8')
    print summary
    print '-'*30
    
def getWords(html):
    # First, strip all html tags off
    txt = re.compile(r'<[^>]+>').sub('', html)
    # Next, split txt by non-alphabetical delimiter
    words = re.compile(r'\W+').split(txt)
    return [w.lower() for w in words if w != '']
    
def getWordCount(url):
    f = feedparser.parse(url)
    wc = defaultdict(int)
    for e in f.entries:
        if 'summary' in e:
            summary = e.summary.encode('utf-8')
        else:
            summary = e.description.encode('utf-8')
        
        words = getWords(e.title.encode('utf-8') + ' ' + summary)
        for word in words:
            wc[word] += 1
    return f.feed.title.encode('utf-8'), wc

def parseFeedList(fname='feedlist.txt', out='data.txt'):
    title2wc = {}
    word2feed = defaultdict(int)
    
    with open(fname, 'rt') as f:
        for line in f:
            try:
                title, wc = getWordCount(line)
                title2wc[title] = wc
                for w, cnt in wc.items():
                    if cnt > 1:
                        word2feed[w] += 1
            except:
                print 'Failed to parse feed %s' % line.rstrip()
                
    wordlist = []
    
    nfeed = len(title2wc.keys())
    
    for w in word2feed:
        frac = float(word2feed[w]) / nfeed
        if frac > 0.1 and frac < 0.5: wordlist.append(w)
    
    with open(out, 'w') as outfile:
        outfile.write('Blog')
        for word in wordlist: outfile.write('\t%s' % word)
        outfile.write('\n')
        for title, wc in title2wc.items():
            outfile.write(title)
            for word in wordlist:
                if word in wc: outfile.write('\t%d' % wc[word])
                else: outfile.write('\t0')
            outfile.write('\n')

def readData(fname='data.txt'):
    with open(fname, 'rt') as f:
        firstLine = f.readline()
        cols = firstLine.strip().split('\t')[1:]
        rows = []
        data = []
        for line in f:
            tmp = line.split('\t')
            rows.append(tmp[0])
            data.append([float(x) for x in tmp[1:]])
    
    return rows, cols, data
    
if __name__ == '__main__':
    #parseFeedList()
    rows, cols, data = readData()
    assert len(rows) == len(data) and len(cols) == len(data[0])