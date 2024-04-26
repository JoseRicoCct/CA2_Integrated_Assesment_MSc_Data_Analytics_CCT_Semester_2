#!/usr/bin/env python3
import sys

# Printing out lines sorted by key (ids)
for line in sys.stdin:
    key, value = line.strip().split("\t", 1)
    print(f"{key},{value}")

