# coding: utf-8

from __future__ import division
from nltk.tokenize import word_tokenize

import nltk
import os
import codecs
import re
import string
import sys

nltk.download('punkt')

NUM = '<NUM>'

PUNCTS = {".": ".PERIOD", "?": "?QUESTIONMARK", ",": ",COMMA"}
IGNORE_TOKENS = set([':', '!', ';', '-'])

forbidden_symbols = re.compile(r"[\[\]\(\)\/\\\>\<\=\+\_\*]")
numbers = re.compile(r"\d")
multiple_punct = re.compile(r'([\.\?\!\,\:\;\-])(?:[\.\?\!\,\:\;\-]){1,}')

SET_UPPER = frozenset(string.uppercase)

is_number = lambda x: len(numbers.sub("", x)) / len(x) < 0.6

def untokenize(line):
    return line.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")

def skip(line):

    if line.strip() == '':
        return True

#    last_symbol = line[-1]
#    if not last_symbol in EOS_PUNCTS:
#        return True

    if forbidden_symbols.search(line) is not None:
        return True

    return False

def process_line(line):

    tokens = word_tokenize(line)
    output_tokens = []

    previspunct = False
    for ii in range(len(tokens)): 
        token = tokens[ii]
        nextisupper = (ii+1 < len(tokens) and tokens[ii+1][0] in SET_UPPER)

        if token in PUNCTS:
            otoken = PUNCTS[token]
            if nextisupper:
                otoken += 'UPPER'
            output_tokens.append(otoken)
            previspunct = True
        elif is_number(token):
            output_tokens.append(NUM)
            previspunct = False
        elif token not in IGNORE_TOKENS:
            if not previspunct and token[0] in SET_UPPER:
                output_tokens.append('^UPPERNEXT')
            output_tokens.append(token.lower())
            previspunct = False

    return untokenize(" ".join(output_tokens) + " ")

skipped = 0

with codecs.open(sys.argv[2], 'w', 'utf-8') as out_txt:
    with codecs.open(sys.argv[1], 'r', 'utf-8') as text:

        for line in text:

            line = line.replace("\"", "").strip()
            line = multiple_punct.sub(r"\g<1>", line)

            if skip(line):
                skipped += 1
                continue

            line = process_line(line)

            out_txt.write(line + '\n')

print "Skipped %d lines" % skipped
