import json
import sys, argparse
from operator import itemgetter
from copy import deepcopy

import time
import random

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


def init_translation_probabilities(corpus):
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


def do_single_iteration(corpus, words, count_english_words, translation_probabilities, cooccurrences):
    counts_cooccurrences = {word_en: {word_fr: 0 for word_fr in possible_words}
                            for word_en, possible_words in cooccurrences.items()}

    occurrences_in_french = {word_fr: 0 for word_fr in words['fr']}

    for (words_in_en, words_in_french) in [(pair['en'].split(), pair['fr'].split())
                                           for pair in corpus]:
        for en_word in words_in_en:
            count_english_words[en_word] = 0

            for fre_w in words_in_french + Null:
                count_english_words[en_word] += translation_probabilities[en_word][fre_w]

        for en_word in words_in_en:
            for fre_w in words_in_french + Null:
                counts_cooccurrences[en_word][fre_w] += (translation_probabilities[en_word][fre_w] /
                                                         count_english_words[en_word])
                occurrences_in_french[fre_w] += translation_probabilities[en_word][fre_w] / count_english_words[en_word]

    for en_word, possible_words in cooccurrences.items():
        for fre_w in possible_words:
            if args.enhanced_smoothing:
                translation_probabilities[en_word][fre_w] = (counts_cooccurrences[en_word][fre_w] + n) / (occurrences_in_french[fre_w] + n *V)
            else:
                translation_probabilities[en_word][fre_w] = counts_cooccurrences[en_word][fre_w] / occurrences_in_french[
                    fre_w]

    return translation_probabilities


def train_model(corpus, iterations_to_preform):
    verbose = args.verbose
    words = get_words_of_both_lang(corpus)

    count_engilsh_words = {word_en: 0 for word_en in words['en']}
    translation_probabilities, cooccurrences = init_translation_probabilities(corpus)

    if verbose:
        print("first iteration started")
    for iteration in range(iterations_to_preform):
        start = time.time()

        translation_probabilities = do_single_iteration(
            corpus, words, count_engilsh_words,
            translation_probabilities, cooccurrences
        )

        end = time.time()
        t = end - start
        if verbose:
            print("iteration %d completed, time took in sec %f, in min %f" % (iteration, t, t / 60))

        if args.output_parameters_every_epoch and args.output_parameter_file_name:
            result_table = prepare_output(translation_probabilities)
            f = args.output_parameter_file_name
            with open(f+"iteration_"+str(iteration), 'w') as f:
                json.dump(result_table, f)

    return translation_probabilities



def create_list_of_sentences(file_a, file_b, less_sentences):
    fa = open(file_a)
    fb = open(file_b)
    a = fa.readlines()
    b = fb.readlines()

    this_fraction = ((len(a) / 100) * less_sentences)
    this_fraction = int(this_fraction)

    final_output = []
    for ii, (e_sentence, f_sentence) in enumerate(zip(a, b)):
        final_output.append({"en": e_sentence, "fr": f_sentence})
        if ii > this_fraction:
            break
    return final_output


def prepare_output(translation_probabilities):
    d = {
        k: sorted(v.items(), key=itemgetter(1), reverse=True)
        for (k, v) in translation_probabilities.items()
    }

    return d


def main():
    english_file_name = args.english_file_name or "../data/hansards.e"
    french_file_name = args.french_file_name or "../data/hansards.f"
    outfile = args.output_parameter_file_name
    iterations = args.number_of_iterations
    debug = args.debug
    less_sentences = args.size
    verbose = args.verbose

    if debug:
        outfile = "debug_output.json"
    if verbose:
        print(outfile)
        print("iterations to perform ", iterations)
        print("less sentences ", less_sentences)
        print("debug ", debug)
        print("output_parameters_every_epoch ", args.output_parameters_every_epoch)
        print("smoothing ", args.enhanced_smoothing)
    corpus = create_list_of_sentences(english_file_name,french_file_name, less_sentences)

    if verbose:
        print("Done reading corpus")

    if debug:
        exit()

    probabilities = train_model(corpus, iterations_to_preform=iterations)
    result_table = prepare_output(probabilities)
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(result_table, f)
    else:
        json.dump(result_table, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--english_file_name", type=str, required=False, default=None)
    parser.add_argument("--french_file_name", type=str, required=False, default=None)
    parser.add_argument("--size", default=100.0, type=float, required=False,
                        help="Fraction to use")
    parser.add_argument("--output_parameter_file_name", type=str, required=True,
                        help="The file name to be generated")

    parser.add_argument("--number_of_iterations", type=int, required=True)
    parser.add_argument("--debug", type=bool, required=False, default=False)
    parser.add_argument("--enhanced_smoothing", type=bool, required=False, default=False)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--output_parameters_every_epoch", type=bool, required=False, default=False)

    global args,n,V
    args = parser.parse_args()
    n = 0.01
    V = 100000

    main()
