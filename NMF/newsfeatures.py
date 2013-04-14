# Chapter 10 - PCI book
import feedparser
import re
import numpy as np
import nmf

feedlist = ['http://rss.cnn.com/rss/edition.rss',
			'http://rss.cnn.com/rss/edition_world.rss',
			'http://rss.cnn.com/rss/edition_us.rss',
			'http://news.google.com/?output=rss',
			'http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
			'http://rss.nytimes.com/services/xml/rss/nyt/.xml',
			'http://feeds2.feedburner.com/time/world',
			'http://feeds.feedburner.com/time/newsfeed',
			'http://feeds2.feedburner.com/time/topstories'
			]
			
def stripHTML(h):
	ret = ''
	s = 0
	for c in h:
		if c == '<': s = 1
		elif c == '>': 
			s = 0
			ret += ' '
		elif s == 0:
			ret += c
	return ret
	
def splitWords(text):
	splitter = re.compile(r'\W*')
	return [s.lower() for s in splitter.split(text) if len(s) > 3]

def getArticleWords():
	allwords = {} # allwords is a dict that allwords[w] = #occurences of w in all docs.
	article_words = [] # list of dicts. Each dict is word<->count mapping of an article. 
	article_titles = [] # list of article titles
	ec = 0
	for feed in feedlist:
		f = feedparser.parse(feed)
		
		for e in f.entries:
			if e.title in article_titles: continue
			txt = e.title.encode('utf-8') + stripHTML(e.description.encode('utf-8'))
			words = splitWords(txt)
			article_words.append({})
			article_titles.append(e.title)
			for word in words:
				allwords.setdefault(word, 0)
				allwords[word] += 1
				article_words[ec].setdefault(word, 0)
				article_words[ec][word] += 1
			ec += 1
			
	return allwords, article_words, article_titles
	
def makeMatrix(allwords, article_words):
	number_of_articles = len(article_words)
	
	# Get words that appear more than 3 times and not appear in more than 60% of the total articles.
	# Intuitively, get words that are general, but not too general.
	wordvec = []
	for w, c in allwords.items():
		if c > 3 and c < number_of_articles * 0.6:
			wordvec.append(w)
			
	mat = [map(lambda x: art[x] if x in art else 0, wordvec) for art in article_words]
	return mat, wordvec
	
def computeSparseness(mat):
	if mat == None: return
	nr = len(mat)
	nc = len(mat[0])
	non_zeros = 0
	for r in range(nr):
		for c in range(nc):
			non_zeros += (1 if mat[r][c] != 0 else 0)
	
	return non_zeros, 100.*non_zeros/nr/nc

def process(mat):
	wMat, fMat = nmf.factorize(np.asarray(mat), pc=20, iter=50)
	return wMat, fMat
	
def showFeatures(w, h, titles, wordvec, out='features.txt', tw=5, ta=3):
	WF = []
	AF = [[] for i in range(len(titles))]
	
	with open(out, 'w') as outfile:
		pc, wc = np.shape(h) # get size of feature vs word matrix
		
		# Loop over features
		for i in range(pc):
			# Top-tw words per feature
			w2f = []
			for j in range(wc):
				w2f.append((h[i,j], wordvec[j]))
			w2f.sort(reverse=True)
			topW = [item[1] for item in w2f[:tw]]
			outfile.write(str(topW)+'\n')
			WF.append(topW)
			
			# Top-ta articles per feature
			a2f = []
			for j in range(len(titles)):
				a2f.append((w[j,i], titles[j]))
				AF[j].append((w[j,i], i, titles[j]))
			a2f.sort(reverse=True)
			for i in range(ta):
				outfile.write(str(a2f[i])+'\n')
			outfile.write('\n')
			
	return WF, AF	

def showArticles(titles, AF, WF, out='articles.txt',top=3):
	with open(out, 'w') as outfile:
		for ai in range(len(titles)):
			# first, print article title
			outfile.write(titles[ai].encode('utf-8') + '\n')
			AF[ai].sort(reverse=True)
			for i in range(top):
				'''
				# get score. Note that AF[ai][i] = (score, feature_idx, article_title)
				score = AF[ai][i][0]
				# get top words for feature. Note that WF[feature_idx] = [word1, word2,...]
				wlist = WF[AF[ai][i][1]]
				# Write to outfile
				outfile.write(str(score) + ' ' + str(wlist) + '\n')
				'''
				outfile.write(str(AF[ai][i][0]) + ' ' + str(WF[AF[ai][i][1]]) + '\n') 
			outfile.write('\n')
	
if __name__ == '__main__':
	allwords, article_words, article_titles = getArticleWords()
	mat, wordvec = makeMatrix(allwords, article_words)
	#_, sparseness = computeSparseness(mat)
	wMat, fMat = process(mat)
	WF, AF = showFeatures(wMat, fMat, article_titles, wordvec)
	showArticles(article_titles, AF, WF)