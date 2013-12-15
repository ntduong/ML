'''
Created on May 6, 2013
@author: Administrator
'''

import json
import sys
from collections import defaultdict

def parseTweets(tweet_file='tweets.txt'):
    parsed_tweets = []
    with open(tweet_file, 'r') as fin:
        for line in fin:
            tweet = json.loads(line)
            if 'text' in tweet:
                parsed_tweets.append(tweet)
    
    return parsed_tweets

def getTags(tweet):
    if 'entities' in tweet:
        if 'hashtags' in tweet['entities']:
            return tweet['entities']['hashtags']

def getAllTags(tweets):
    hashtags = defaultdict(float)
    for tweet in tweets:
        tags = getTags(tweet)
        if tags:
            for item in tags:
                hashtags[item['text']] += 1.0
    return hashtags

def main():
    tweets = parseTweets(sys.argv[1])
    hashtags = getAllTags(tweets)
    cnt = 0
    for t in sorted(hashtags, key=hashtags.get, reverse=True):
        if cnt == 10: break
        print t, hashtags[t]
        cnt += 1
    
if __name__ == '__main__':
    main()