#!/usr/bin/env python3
import sys
import csv

def clean_text(text):
    # Replace or escape the commas. Here we're just removing them.
    return text.replace(',', '')

for line in sys.stdin:
    # Remove any leading or trailing whitespace
    line = line.strip()
    
    # Parse the line using csv.reader which handles commas inside quotes
    try:
        columns = next(csv.reader([line], delimiter=',', quotechar='"'))
        # Assuming the 'text' column is the last one
        text_col_index = len(columns) - 1
        # Clean the 'text' column
        columns[text_col_index] = clean_text(columns[text_col_index])
        # Output the cleaned row
        print(','.join(columns))
    except csv.Error as e:
        # Log or handle error for this line
        sys.stderr.write("Error parsing line: {}\n".format(line))

