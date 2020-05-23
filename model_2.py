import json
import sys, argparse
from operator import itemgetter
import numpy as np
import time
import random
from collections import defaultdict

Null = [None]


def vector_sums_to_1(v):
    s = sum(v)
    r = [i / s for i in v]
    return r


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


def get_set_of_length_pairs(corpus, use_set=True):
    if use_set:
        set_of_pairs = set()
    else:
        list_of_pairs = []
    for pair in corpus:
        en_length = len(pair['en'].split())
        fr_length = len(pair['fr'].split())
        if use_set:
            set_of_pairs.add((en_length, fr_length))
        else:
            list_of_pairs.append((en_length, fr_length))
    if use_set:
        return set_of_pairs
    else:
        return list_of_pairs


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


def read_alignments(file):
    all = []
    with open(file) as fr:
        for line in fr:
            alignments_of_line = set([tuple(map(int, x.split("-"))) for x in line.strip().split()])
            a = defaultdict(lambda: -1, {e: f for f, e in alignments_of_line})
            all.append(a)
    return all


def get_init_alignments_from_model_1(file_to_read, corpus):
    previous_alignments = read_alignments(file_to_read)
    count_e = defaultdict(lambda: 0, {})
    count_alingment = defaultdict(lambda: 0, {})
    list_of_pairs = get_set_of_length_pairs(corpus=corpus, use_set=False)
    for pair, prev in zip(list_of_pairs, previous_alignments):
        en_length = pair[0]
        fr_length = pair[1]
        for en_word_index in range(en_length):
            fr_index = prev[en_word_index]
            if fr_index == -1: fr_index = fr_length
            count_alingment[fr_index, en_word_index, fr_length, en_length] += 1
            count_e[en_word_index, fr_length, en_length] += 1
    a = {k: (v / (count_e[k[1:]])) for k, v in count_alingment.items()}
    return defaultdict(lambda: 10 ** -2, a)


def init_alignments_probabilities(corpus):
    if args.initialization_file:
        file_to_read = args.initialization_file
        print(file_to_read)
        return get_init_alignments_from_model_1(file_to_read, corpus)

    set_of_pairs = get_set_of_length_pairs(corpus)
    a = {}
    for pair in set_of_pairs:
        en_length = pair[0]
        fr_length = pair[1]
        for en_word_index in range(en_length):
            x = [random.random() for fr_index in range(fr_length + 1)]
            x = vector_sums_to_1(x)
            for fr_index, p in enumerate(x):
                a[fr_index, en_word_index, fr_length, en_length] = p

    return a


def do_single_iteration(corpus, translation_probabilities, cooccurrences, align):
    count_cooccurrences = defaultdict(float)

    count_english_words = defaultdict(float)
    count_alignments = defaultdict(float)
    count_total_into_index_i = defaultdict(float)

    for (words_in_en, words_in_french) in [(pair['en'].split(), pair['fr'].split())
                                           for pair in corpus]:
        length_en = len(words_in_en)
        length_fr = len(words_in_french)
        for en_index, en_word in enumerate(words_in_en):
            nominator = []
            for fr_index, fre_w in enumerate(words_in_french + Null):
                nominator.append(
                    align[fr_index, en_index, length_fr, length_en] * translation_probabilities[en_word][fre_w])

            denominator = sum(nominator)

            for fr_index, fre_w in enumerate(words_in_french + Null):
                delta = nominator[fr_index] / denominator
                count_cooccurrences[fre_w, en_word] += delta
                count_english_words[fre_w] += delta
                count_alignments[fr_index, en_index, length_fr, length_en] += delta
                count_total_into_index_i[en_index, length_fr, length_en] += delta

    for en_word, possible_words in cooccurrences.items():
        for fre_w in possible_words:
            translation_probabilities[en_word][fre_w] = count_cooccurrences[fre_w, en_word] / count_english_words[fre_w]
    align = {}
    for k, v in count_alignments.items():
        align[k] = v / count_total_into_index_i[k[1:]]

    return translation_probabilities, align


def train_model(corpus, iterations_to_preform):
    verbose = args.verbose

    translation_probabilities, cooccurrences = init_translation_probabilities(corpus)
    alignments = init_alignments_probabilities(corpus)

    # alignments of q(j|iii,l.m) use alignments[length_of_english_sentence,length_of_french_sentence][french_index]

    if verbose:
        print("first iteration started")
    for iteration in range(iterations_to_preform):
        start = time.time()

        translation_probabilities, alignments = do_single_iteration(
            corpus, translation_probabilities, cooccurrences, alignments
        )

        end = time.time()
        t = end - start
        if verbose:
            print("iteration %d completed, time took in sec %f, in min %f" % (iteration, t, t / 60))

        if args.output_parameters_every_epoch and args.output_parameter_file_name:
            result_table = prepare_output(translation_probabilities, alignments)
            f = args.output_parameter_file_name
            with open(f + "iteration_" + str(iteration), 'w') as f:
                json.dump(result_table, f)

    return translation_probabilities, alignments


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


def prepare_output(translation_probabilities, alignments):
    combined = {"trans": {
        k: sorted(v.items(), key=itemgetter(1), reverse=True)
        for (k, v) in translation_probabilities.items()
    },
        "alignments": remap_keys(alignments)
    }

    return combined


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
    corpus = create_list_of_sentences(english_file_name, french_file_name, less_sentences)

    if verbose:
        print("Done reading corpus")

    probabilities, alignments = train_model(corpus, iterations_to_preform=iterations)

    result_table = prepare_output(probabilities, alignments)
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(result_table, f)
    else:
        json.dump(result_table, sys.stdout)


def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--english_file_name", type=str, required=False, default=None)
    parser.add_argument("--french_file_name", type=str, required=False, default=None)
    parser.add_argument("--initialization_file", type=str, required=False, default=None)

    parser.add_argument("--size", default=100.0, type=float, required=False,
                        help="Fraction to use")

    parser.add_argument("--output_parameter_file_name", type=str, required=True,
                        help="The file name to be generated")

    parser.add_argument("--number_of_iterations", type=int, required=True)
    parser.add_argument("--debug", type=bool, required=False, default=False)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--output_parameters_every_epoch", type=bool, required=False, default=False)

    global args
    args = parser.parse_args()

    main()
