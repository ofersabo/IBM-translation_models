import json
import sys, argparse
from operator import itemgetter
from copy import deepcopy

import time


def get_sentences(filename):
    with open(filename, 'r') as f:
        corpus = json.load(f)

    return corpus


def get_words_of_both_lang(corpus):
    def source_words(lang):
        for pair in corpus:
            for word in pair[lang].split():
                yield word

    return {lang: set(source_words(lang)) for lang in ('en', 'fr')}


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
            for fr_word in pair['fr'].split():
                this_set.add(fr_word)
    initial_translation = {wr_en: {fr_word: (1 / len(cooccurrences[wr_en])) for fr_word in cooccurrences[wr_en]} for
                           wr_en in cooccurrences}

    return initial_translation, cooccurrences
    # return {
    #     word_en: {word_fr: 1/len(words['en'])
    #               for word_fr in words['fr']}
    #     for word_en in words['en']}


def train_iteration(corpus, words, count_english_words, prev_translation_probabilities, cooccurrences):
    '''
    Perform one iteration of the EM-Algorithm

    corpus: corpus object to train from
    words: {language: {word}} mapping

    count_english_words: counts of the destination words, weighted according to
             their translation probabilities t(e|s)

    '''
    translation_probabilities = prev_translation_probabilities

    counts_cooccurrences = {word_en: {word_fr: 0 for word_fr in possible_words}
                            for word_en, possible_words in cooccurrences.items()}

    occurrences_in_french = {word_fr: 0 for word_fr in words['fr']}

    for (words_in_en, words_in_french) in [(pair['en'].split(), pair['fr'].split())
                                           for pair in corpus]:
        for en_word in words_in_en:
            count_english_words[en_word] = 0

            for fre_w in words_in_french:
                count_english_words[en_word] += translation_probabilities[en_word][fre_w]

        for en_word in words_in_en:
            for fre_w in words_in_french:
                counts_cooccurrences[en_word][fre_w] += (translation_probabilities[en_word][fre_w] /
                                                         count_english_words[en_word])
                occurrences_in_french[fre_w] += translation_probabilities[en_word][fre_w] / count_english_words[en_word]

    for en_word, possible_words in cooccurrences.items():
        for fre_w in possible_words:
            translation_probabilities[en_word][fre_w] = counts_cooccurrences[en_word][fre_w] / occurrences_in_french[
                fre_w]

    return translation_probabilities


def train_model(corpus, iterations_to_preform):
    '''
    Given a `corpus`, train a translation model on that corpus
    '''
    verbose = args.verbose
    words = get_words_of_both_lang(corpus)

    count_engilsh_words = {word_en: 0 for word_en in words['en']}
    inital_translation_probabilities, cooccurrences = init_translation_probabilities(corpus)

    # iterations = 0
    if verbose:
        print("first iteration started")
    for iteration in range(iterations_to_preform):
        start = time.time()

        translation_probabilities = train_iteration(
            corpus, words, count_engilsh_words,
            inital_translation_probabilities, cooccurrences
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

    probabilities = train_model(corpus, iterations_to_preform=iterations)
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

    global args
    args = parser.parse_args()

    main()
