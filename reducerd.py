#!/usr/bin/env python3

import sys

# Counting unique tweet_id values
current_tweet_id = None
unique_count = 0

for line in sys.stdin:
    # Strip any leading and trailing whitespace
    line = line.strip()
    # Parse the input from the mapper
    tweet_id, count = line.split('\t')

    # If the current tweet_id is different from the previous one,
    # it's a new unique tweet_id
    if current_tweet_id != tweet_id:
        unique_count += 1
        current_tweet_id = tweet_id

# Printing count of unique tweet_ids
print(f"Unique tweet_id count: {unique_count}")

