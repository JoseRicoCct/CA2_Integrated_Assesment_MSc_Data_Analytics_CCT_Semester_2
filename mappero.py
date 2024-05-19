#!/usr/bin/env python3
import sys

# Mapper outputs the ids as key and the rest of the columns as value
for line in sys.stdin:
    line = line.strip()
    parts = line.split(',', 1)  # Split the line into key (id) and value (the rest)
    if len(parts) == 2:
        print(f"{parts[0]}\t{parts[1]}")

