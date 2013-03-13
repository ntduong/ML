# Simple feed filter
import feedparser
import re

def read(feedURL, classifier):
	f = feedparser.parse(feedURL)
	for entry in f['entries']:
		print
		print '-' * 30
		print 'Title: ' + entry['title'].encode('utf-8')
		print 'Publisher: ' + entry['publisher'].encode('utf-8')
		print
		print entry['summary'].encode('utf-8')
		
		fulltext = '%s\n%s\n%s' %(entry['title'], entry['publisher'], entry['summary'])
		print 'Guess: ' + str(classifier.classify(fulltext))
		
		# Ask user to provide his category for this feed, and use this to train classifier.
		userAnswer = raw_input('Enter category: ')
		classifier.train(fulltext, userAnswer)

