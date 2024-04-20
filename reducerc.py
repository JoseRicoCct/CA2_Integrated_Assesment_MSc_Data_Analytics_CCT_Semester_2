#!/usr/bin/env python3
import sys

# Simply read from standard input and output the line as is
for line in sys.stdin:
    # Since we are assuming the mapper already cleaned the lines,
    # the reducer can just output them directly.
    print(line.strip())

