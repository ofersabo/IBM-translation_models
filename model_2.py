import json
import sys, argparse
from operator import itemgetter
from copy import deepcopy
import numpy as np
import time
import random
from collections import defaultdict


def vector_sums_to_1(v):
    s = sum(v)
    r = [i / s for i in v]
    return r


Null = [None]


def get_sentences(filename):
    with open(filename, 'r') as f:
        corpus = json.load(f)

    return corpus


def get_words_of_both_lang(corpus):
    def source_words(lang):
        for pair in corpus:
            for word in pair[lang].split():
                yield word

    d = {lang: set(source_words(lang)) for lang in ('en', 'fr')}
    d['fr'].update(Null)
    return d


def get_maximal_length(corpus):
    maximal_en = 0
    maximal_fr = 0
    for pair in corpus:
        en_length = len(pair['en'].split())
        fr_length = len(pair['fr'].split())
        if en_length > maximal_en:
            maximal_en = en_length
        if fr_length > maximal_fr:
            maximal_fr = fr_length

    return maximal_en, maximal_fr


def init_translation_probabilities(corpus):
    '''
    Given a `corpus` generate the first set of translation probabilities,
    which can be accessed as
    p(e|s) <=> translation_probabilities[e][s]
    we first assume that for an `e` and set of `s`s, it is equally likely
    that e will translate to any s in `s`s
    '''
    words = get_words_of_both_lang(corpus)
    cooccurrences = {wr_en: set() for wr_en in words['en']}
    for pair in corpus:
        for wr_en in pair['en'].split():
            this_set = cooccurrences[wr_en]
            this_set.update(Null)
            for fr_word in pair['fr'].split():
                this_set.add(fr_word)
    initial_translation = {wr_en: {fr_word: (1 / len(cooccurrences[wr_en])) for fr_word in cooccurrences[wr_en]} for
                           wr_en in cooccurrences}

    return initial_translation, cooccurrences


def init_alignments_probabilities(corpus):
    maximal_a, maximal_b = get_maximal_length(corpus)
    a = {}
    for en_length in range(maximal_a):
        for fr_length in range(maximal_b + 1):
            x = [random.random() for i in range(fr_length)]
            x = vector_sums_to_1(x)
            a[en_length, fr_length] = x

    return a


def train_iteration(corpus, words, count_english_words, translation_probabilities, cooccurrences, align):
    '''
    Perform one iteration of the EM-Algorithm
    count_english_words: counts of the destination words, weighted according to
             their translation probabilities t(e|s)
    '''

    counts_cooccurrences = {word_en: {word_fr: 0 for word_fr in possible_words}
                            for word_en, possible_words in cooccurrences.items()}

    occurrences_in_french = {word_fr: 0 for word_fr in words['fr']}

    count_from_word_j_to_word_i = {k: [0] * len(v) for k,v in align}
    sum_of_outgoing_from_index_in_french = {k: [0] * len(v) for k,v in align}
    # alignments = defaultdict(float)  # wj aligned with wi
    # c4 = defaultdict(float)  # wi aligned with anything

    # alignments of q(j|i,l,m) use alignments[length_of_english_sentence,length_of_french_sentence][french_index]

    for (words_in_en, words_in_french) in [(pair['en'].split(), pair['fr'].split())
                                           for pair in corpus]:
        length_en = len(words_in_en)
        length_fr = len(words_in_french) + 1
        delta_for_all_words = []
        this_single_sen_count_alignment_from_index_i = []
        for en_index, en_word in enumerate(words_in_en):
            count_english_words[en_word] = 0
            nominator = []
            for fr_index, fre_w in enumerate(words_in_french + Null):
                this_single_sen_count_alignment_from_index_i.append(0)
                nominator.append(align[length_en, length_fr][fr_index] * translation_probabilities[en_word][fre_w])

            denominator = sum(nominator)
            delta = [t / denominator for t in nominator]
            delta_for_all_words.append(delta)

            for fr_index, fre_w in enumerate(words_in_french + Null):
                count_english_words[en_word] += delta[fr_index]
                this_single_sen_count_alignment_from_index_i[fr_index] += delta[fr_index]

        for en_index, en_word in enumerate(words_in_en):
            for fr_index, fre_w in enumerate(words_in_french + Null):
                counts_cooccurrences[en_word][fre_w] += (delta_for_all_words[en_index][fr_index] /
                                                         count_english_words[en_word])

                occurrences_in_french[fre_w] += delta_for_all_words[en_index][fr_index] / count_english_words[en_word]

                count_from_word_j_to_word_i[length_en,length_fr] += delta_for_all_words[en_index][fr_index] / \
                                                                   this_single_sen_count_alignment_from_index_i[
                                                                       fr_index]
                sum_of_outgoing_from_index_in_french[fr_index] += delta_for_all_words[en_index][fr_index]


    for en_word, possible_words in cooccurrences.items():
        for fre_w in possible_words:
            translation_probabilities[en_word][fre_w] = counts_cooccurrences[en_word][fre_w] / occurrences_in_french[
                fre_w]

    return translation_probabilities, alignments


def train_model(corpus, iterations_to_preform):
    '''
    Given a `corpus`, train a translation model on that corpus
    '''
    verbose = args.verbose
    words = get_words_of_both_lang(corpus)

    count_engilsh_words = {word_en: 0 for word_en in words['en']}
    inital_translation_probabilities, cooccurrences = init_translation_probabilities(corpus)
    alignments = init_alignments_probabilities(corpus)

    # alignments of q(j|i,l.m) use alignments[length_of_english_sentence,length_of_french_sentence][french_index]

    # iterations = 0
    if verbose:
        print("first iteration started")
    for iteration in range(iterations_to_preform):
        start = time.time()

        translation_probabilities, alignments = train_iteration(
            corpus, words, count_engilsh_words,
            inital_translation_probabilities, cooccurrences, alignments
        )

        end = time.time()
        t = end - start
        if verbose:
            print("iteration %d completed, time took in sec %f, in min %f" % (iteration, t, t / 60))
        # prev_translation_probabilities = translation_probabilities
        # iterations += 1
        # with open(translation_probabilities_file,"w") as fw:
        #     json.dump(translation_probabilities,fw)
    return translation_probabilities


# def get_alignments(translations,source,target):
#     words_alignments = []
#     for i in range(len(target)):


def create_list_of_sentences(file_a, file_b, less_sentences):
    fa = open(file_a)
    fb = open(file_b)
    a = fa.readlines()
    b = fb.readlines()

    this_fraction = ((len(a) / 100) * less_sentences)
    this_fraction = int(this_fraction)

    print(type(this_fraction))
    print(this_fraction)

    final_output = []
    for ii, (e_sentence, f_sentence) in enumerate(zip(a, b)):
        final_output.append({"en": e_sentence, "fr": f_sentence})
        if ii > this_fraction:
            break
    return final_output


def prepare_output(translation_probabilities):
    d = {
        # for each english word
        # sort the words it could translate to; most probable first
        k: sorted(v.items(), key=itemgetter(1), reverse=True)
        # then grab the head of that == `(most_probable, p(k|most probable)`
        # and the first of that pair (the actual word!)
        for (k, v) in translation_probabilities.items()
    }
    # for en_w,fr_w_and_prob in d.items():
    #     compute_ratio = fr_w_and_prob[0][1] / 100
    #     for ii,(w,p) in enumerate(fr_w_and_prob):
    #         if p < compute_ratio and ii>3:
    #             break
    #     d[en_w] = fr_w_and_prob[:ii]

    return d


def main():
    infile = args.file_name
    outfile = args.output_parameter_file_name
    iterations = args.number_of_iterations
    debug = args.debug
    less_sentences = args.size
    verbose = args.verbose

    if debug:
        if not infile:
            infile = "data/sentences.json"
        outfile = "debug_output.json"
    if verbose:
        print(outfile)
        print("iterations to perform ", iterations)
        print("less sentences ", less_sentences)
        print("debug ", debug)
    if infile:
        corpus = get_sentences(infile)
    else:
        corpus = create_list_of_sentences("../data/hansards.e", "../data/hansards.f", less_sentences)
    if verbose:
        print("Done reading corpus")

    probabilities, alignments = train_model(corpus, iterations_to_preform=iterations)

    result_table = prepare_output(probabilities)
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(result_table, f)
    else:
        json.dump(result_table, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=False, default=None)
    parser.add_argument("--size", default=100.0, type=float, required=False,
                        help="Fraction to use")
    parser.add_argument("--output_parameter_file_name", type=str, required=True,
                        help="The file name to be generated")

    parser.add_argument("--number_of_iterations", type=int, required=True)
    parser.add_argument("--debug", type=bool, required=False, default=False)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--model-2", type=bool, required=False, default=True)

    global args

    args = parser.parse_args()

    main()
