#!/usr/bin/env python3
import sys

for line in sys.stdin:
    # Strip any leading and trailing whitespace from the line
    line = line.strip()
    # Split the row by comma and get the first five fields plus the rest of the text
    parts = line.split(',', 5)
    # Ensure that we have at least six parts; if not, add empty fields
    while len(parts) < 6:
        parts.append('')
    # For the sixth field, remove any remaining commas and
    # also removing quotes they are breaking Cassandra
    parts[5] = parts[5].replace(',', '').replace('"', '')
    # For the sixth field, 
    parts[5] = parts[5].replace(',', '')
    # Join all parts together, last part for column text
    cleaned_line = ','.join(parts)
    # Output the cleaned line
    print(cleaned_line)

