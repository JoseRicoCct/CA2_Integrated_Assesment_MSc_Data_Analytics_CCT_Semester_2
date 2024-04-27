#!/usr/bin/env python3

import sys

# Passing tweet_id values
for line in sys.stdin:
    # Strip leading and trailing whitespace
    line = line.strip()
    # Splitting the line into columns
    parts = line.split(',')
    if len(parts) > 1:
        # Emit the tweet_id
        print(f"{parts[1]}\t1")

