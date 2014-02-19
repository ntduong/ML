# Simple feed filter
import feedparser
import re
import textclassifier

def entryFeatures(entry):
    splitter = re.compile(r'\W+')
    f = []
    
    titlewords = [s.lower() for s in splitter.split(entry.title) if len(s) > 2 and len(s) < 20]
    
    for w in titlewords:
        f.append('Title:'+w)
    
    if 'summary' in entry:
        summary = entry.summary.encode('utf-8')
    else:
        summary = entry.description.encode('utf-8')
    
    summarywords = [s for s in splitter.split(summary) if len(s) > 2 and len(s) < 20]
    uc = 0
    for w in summarywords:
        f.append(w.lower())
        if w.isupper(): uc += 1
    
    if float(uc)/len(summarywords) > 0.3:
        f.append('UPPERCASE')
        
    for i in range(len(summarywords)-1):
        twowords = ' '.join(summarywords[i:i+2])
        f.append(twowords.lower())
    
    f.append('Author:' + entry.author.encode('utf-8'))
    
    return set(f)
    
def readFeed(feedURL, classifier):
    f = feedparser.parse(feedURL)
    for entry in f.entries:
        print
        print '-' * 30
        print 'Title: ' + entry.title.encode('utf-8')
        print 'Author: ' + entry.author.encode('utf-8')
        print
        print entry.summary.encode('utf-8')
        
        #fulltext = '%s\n%s\n%s' %(entry.title.encode('utf-8'), entry.author.encode('utf-8'), \
        #entry.summary.encode('utf-8'))
        #print 'Guess: ' + str(classifier.classify(fulltext))
        print 'Guess: ' + str(classifier.classify(entry))
        
        # Ask user to provide his category for this feed, and use this to train classifier.
        userAnswer = raw_input('Enter category: ')
        classifier.train(entry, userAnswer)

def test(fname='feedlist.txt'):
    aClassifier = textclassifier.NBClassifier(entryFeatures)
    with open(fname, 'rt') as f:
        for line in f:
            readFeed(line.strip(), aClassifier)
    
if __name__ == '__main__':
    test()