#!/usr/bin/env python
# coding: utf-8
#Written by Desmond Caulley

# This file takes a rough list of celebrity names and returns a 
# cleaner version of the list. Two arguments need to be 
# supplied - input file diretory and output file directory.

# Eg. usage: clean_celeb_list.py input_file.txt output_file.txt

import sys
import numpy as np
from string import digits, punctuation

from fuzzywuzzy import fuzz #pip install fuzzywuzzy
from fuzzywuzzy import process 

input_dir = sys.argv[1]
output_dir = sys.argv[2]

f_read = open(input_dir, "r")
data = f_read.read().split('\n')
f_read.close()


#parsing
data_rev = []
for c, celeb_name in enumerate(data):
    if '(' in celeb_name:
        data[c] = data[c][: data[c].find('(')]

data = [k.lstrip(digits).rstrip(digits).strip(punctuation).strip().strip(digits) for k in data]
data = list(filter(bool, data))

#unique names
data = list(set(data))
data.sort()


# Removing duplicates which slightly differnt spellings
for k in range(len(data)-1):
    if fuzz.ratio(data[k], data[k+1]) > 70:
        data[k] = ""

data = list(filter(bool, data))


# Write new cleaner file
with open(output_dir, 'w') as f:
    for d in data:
        f.write(d + '\n')



