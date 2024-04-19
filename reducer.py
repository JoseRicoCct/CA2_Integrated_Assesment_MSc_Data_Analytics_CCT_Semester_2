#!/usr/bin/env python3

import sys

def print_sorted_counts(sorted_counts, title):
    print(title)
    for entity, count in sorted_counts:
        print(f'{entity}\t{count}')

hashtag_counts = {}
mention_counts = {}

# Read from STDIN
for line in sys.stdin:
    line = line.strip()
    entity, count = line.split('\t', 1)
    count = int(count)
    
    if entity.startswith('H'):
        hashtag = entity[1:]  # Remove the identifier
        hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + count
    elif entity.startswith('M'):
        mention = entity[1:]  # Remove the identifier
        mention_counts[mention] = mention_counts.get(mention, 0) + count

# Filter and sort hashtags and mentions
filtered_hashtags = {hashtag: count for hashtag, count in hashtag_counts.items() if count >= 50}
filtered_mentions = {mention: count for mention, count in mention_counts.items() if count >= 50}

sorted_hashtags = sorted(filtered_hashtags.items(), key=lambda item: item[1], reverse=True)
sorted_mentions = sorted(filtered_mentions.items(), key=lambda item: item[1], reverse=True)

# Print sorted and filtered results
print_sorted_counts(sorted_hashtags, "Hashtags Ranking:")
print()
print_sorted_counts(sorted_mentions, "Mentions Ranking:")

