#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import print_function

import argparse
import codecs
import logging
import sys

import theano

import numpy as np
import theano.tensor as T

import models
import data


MAX_SUBSEQUENCE_LEN = 200

VERSION = 0.1

class Global:
    '''Stores globals. There should be no instances of Global.'''

    # Command line arguments
    args = None

# end class Global


def main(argv):
    parse_arguments(argv[1:])
    setup_logging()

    x = T.imatrix('x')
    
    logging.info("Loading model parameters...")
    net, _ = models.load(Global.args.model_file, 1, x)
    logging.info("Building model...")
    predict = theano.function(
        inputs=[x],
        outputs=net.y
    )

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

    for input_text in sys.stdin:
        text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
        print(restore(text, word_vocabulary, reverse_punctuation_vocabulary, predict))

# end main()


def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T
# end to_array()


def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]
# end convert_punctuation_to_readable()

    
def restore(text, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    ret = []
    while True:

        subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

        y = predict_function(to_array(converted_subsequence))

        ret.append(subsequence[0])

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(y_t.flatten())
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            ret.append(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
            if j < step - 1:
                ret.append(subsequence[1+j])

        if subsequence[-1] == data.END:
            break

        i += step
    return ''.join(ret)
# end restore()


def parse_arguments(strs):
    parser = argparse.ArgumentParser(description='Description. Version %s.'%(VERSION))
    parser.add_argument('-loglevel',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='WARNING',
                        help='Logging level (default %(default)s)')
    parser.add_argument('-version', '--version', action='version', version=str(VERSION))
    parser.add_argument('model_file', help='Input model file')
    Global.args = parser.parse_args(strs)
# end parse_arguments()


def setup_logging():
    numeric_level = getattr(logging, Global.args.loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % Global.args.loglevel)
    logging.basicConfig(level=numeric_level, format="%(module)s:%(levelname)s: %(message)s")
# end setup_logging()


if __name__ == "__main__":
    main(sys.argv)
