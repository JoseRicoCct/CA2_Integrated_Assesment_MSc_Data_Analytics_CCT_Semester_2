#!/usr/bin/env python3

import sys
import re

def extract_entities(text):
    # Match hashtags and mentions
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    return hashtags, mentions

for line in sys.stdin:
    line = line.strip()
    tweet_text = line.split('\t')[-1]
    hashtags, mentions = extract_entities(tweet_text)
    for hashtag in hashtags:
        print(f'H{hashtag}\t1')
    for mention in mentions:
        print(f'M{mention}\t1')

