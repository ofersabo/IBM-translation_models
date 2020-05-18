import json
import pprint
import clize
import sys
from operator import itemgetter
from copy import deepcopy


VERBOSE = False

def distance(table_1, table_2):
    '''
    modelling the tables as vectors, whose indices are essentially some
    hashing function applied to each (row key, col key) pair, return the
    euclidean distance between them, where euclidean distance is defined as

    sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + ... + (a[n] - b[n])**2)

    assumes that table_1 and table_2 are identical in structure
    '''
    row_keys = table_1.keys()
    cols = list(table_1.values())
    col_keys = cols[0].keys()

    result = 0
    for (row_key, col_key) in zip(row_keys, col_keys):
        delta = (table_1[row_key][col_key] -
                 table_2[row_key][col_key]) ** 2
        result += delta

    return result ** 0.5




def get_corpus(filename):
    '''
    Load corpus file located at `filename` into a list of dicts
    '''
    with open(filename, 'r') as f:
        corpus = json.load(f)

    # if VERBOSE:
    #     print(corpus, file=sys.stderr)
    return corpus

def get_words_of_both_lang(corpus):
    '''
    From a `corpus` object, build a dict whose keys are 'en' and 'fr',
    and whose values are sets. Each dict[language] set contains every
    word in that language which appears in the corpus
    '''
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
    return {
        word_en: {word_fr: 1/len(words['en'])
                  for word_fr in words['fr']}
        for word_en in words['en']}


def train_iteration(corpus, words, count_english_words, prev_translation_probabilities):
    '''
    Perform one iteration of the EM-Algorithm

    corpus: corpus object to train from
    words: {language: {word}} mapping

    count_english_words: counts of the destination words, weighted according to
             their translation probabilities t(e|s)

    prev_translation_probabilities: the translation_probabilities from the
                                    last iteration of the EM algorithm
    '''
    translation_probabilities = deepcopy(prev_translation_probabilities)

    counts_cooccurrences = {word_en: {word_fr: 0 for word_fr in words['fr']}
              for word_en in words['en']}

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

    for fre_w in words['fr']:
        for en_word in words['en']:
            translation_probabilities[en_word][fre_w] = counts_cooccurrences[en_word][fre_w] / occurrences_in_french[fre_w]

    return translation_probabilities


def is_converged(probabilties_prev, probabilties_curr, epsilon):
    '''
    Decide when the model whose final two iterations are
    `probabilties_prev` and `probabilties_curr` has converged
    '''
    delta = distance(probabilties_prev, probabilties_curr)
    if VERBOSE:
        print(delta, file=sys.stderr)

    return delta < epsilon


def train_model(corpus, epsilon,translation_probabilities_file):
    '''
    Given a `corpus` and `epsilon`, train a translation model on that corpus
    '''
    words = get_words_of_both_lang(corpus)

    count_engilsh_words = {word_en: 0 for word_en in words['en']}
    prev_translation_probabilities = init_translation_probabilities(corpus)

    converged = False
    iterations = 0
    while not converged:
        translation_probabilities = train_iteration(
                                        # this is a disgusting way
                                        # to indent code
                                        corpus, words, count_engilsh_words,
                                        prev_translation_probabilities
                                    )

        converged = is_converged(prev_translation_probabilities,
                                 translation_probabilities, epsilon)
        prev_translation_probabilities = translation_probabilities
        iterations += 1
        # with open(translation_probabilities_file,"w") as fw:
        #     json.dump(translation_probabilities,fw)
    return translation_probabilities, iterations


# def get_alignments(translations,source,target):
#     words_alignments = []
#     for i in range(len(target)):



def create_list_of_sentences(file_a,file_b,debug):
    fa = open(file_a)
    fb = open(file_b)
    a = fa.readlines()
    b = fb.readlines()
    final_output = []
    for ii,(e_sentence,f_sentence) in enumerate(zip(a,b)):
        final_output.append({"en":e_sentence,"fr":f_sentence})
        if debug and ii >1000:
            break
    return final_output

def summarize_results(translation_probabilities):
    '''
    from a dict of source: {target: p(source|target}, return
    a list of mappings from source words to the most probable target word
    '''
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

def main(infile = None, *, outfile:'o'=None, epsilon:'e'=0.005, verbose:'v'=True,debug=True,less_sentences=False):
    '''
    IBM Model 1 SMT Training Example

    infile: path to JSON file containing English-French sentence pairs
            in the form [ {"en": <sentence>, "fr": <sentence>}, ... ]

    outfile: path to output file (defaults to stdout)

    epsilon: Acceptable euclidean distance between translation probability
             vectors across iterations

    verbose: print running info to stderr
    '''
    translation_probabilities_file = "translation_probabilities.json"
    if debug:
        if not infile:
            infile = "data/sentences.json"
        outfile = "debug_output.json"
        translation_probabilities_file = "debug_"+translation_probabilities_file

    print(outfile)
    print("less sentences ", less_sentences)

    global VERBOSE
    VERBOSE = verbose
    if infile:
        corpus = get_corpus(infile)
    else:
        corpus = create_list_of_sentences("../data/hansards.e", "../data/hansards.f",less_sentences)

    probabilities, iterations = train_model(corpus, epsilon,translation_probabilities_file)
    result_table = summarize_results(probabilities)
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(result_table, f)
    else:
        json.dump(result_table, sys.stdout)

    if VERBOSE:
        print('Performed {} iterations'.format(iterations), file=sys.stderr)

if __name__ == '__main__':
    clize.run(main)
