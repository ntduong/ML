'''
Created on May 4, 2013
@author: Administrator
'''

import json
import sys

def parseTweets(tweet_file='tweets.txt'):
    parsed_tweets = []
    with open(tweet_file, 'r') as fin:
        for line in fin:
            tweet = json.loads(line)
            if 'text' in tweet:
                parsed_tweets.append(tweet)
    
    return parsed_tweets
        
def readSentimentFile(filename='AFINN-111.txt'):
    term2score = {}
    with open(filename, 'r') as fin:
        for line in fin:
            term, score = line.strip().rsplit('\t', 1)
            score = float(score)
            term2score[term] = score
            
    return term2score
        
def scoreTweet(tweet, term2score):
    text = tweet['text']
    terms = text.split()
    scores = map(lambda t: term2score.get(t, float(0)), terms)
    return sum(scores)

def main():
    term2score = readSentimentFile(sys.argv[1])
    tweets = parseTweets(sys.argv[2])
    for tweet in tweets:
        print scoreTweet(tweet, term2score)
        
if __name__ == '__main__':
    main()
    