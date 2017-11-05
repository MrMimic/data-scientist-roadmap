#!/usr/bin/python3

# Text coming from Python module __re__
text = 'This module provides regular expression matching operations similar to those found in Perl.'

# import re library
import re

# Substitution of "Perl" by "every languages"
new_text = re.sub('Perl', 'every languages', text)
print(new_text)

# Searching for capitals letters in the text
new_text = re.findall('[A-Z]', text)
print(new_text)

# Test if a word is in the text or not
new_text = re.match('.*regular.*', text)
print(new_text)
