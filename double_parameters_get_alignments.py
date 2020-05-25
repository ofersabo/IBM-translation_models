import json
import sys
from collections import defaultdict

def create_list_of_sentences(english_f, french_f, debug):
    fa = open(english_f)
    fb = open(french_f)
    a = fa.readlines()
    b = fb.readlines()
    final_output = []
    for ii,(e_sentence,f_sentence) in enumerate(zip(a,b)):
        final_output.append({"en":e_sentence,"fr":f_sentence})
        if debug and ii >1000:
            break
    return final_output


def get_corpus(filename):
    '''
    Load corpus file located at `filename` into a list of dicts
    '''
    with open(filename, 'r') as f:
        corpus = json.load(f)

    # if VERBOSE:
    #     print(corpus, file=sys.stderr)
    return corpus

def build_alignments(parameters, corpus,second_parameters):
    align = defaultdict(lambda: 10 ** -5, {})
    if len(parameters) == 2 and "alignments" in parameters:
        print("output alignment from model 2 parameters")
        translations = parameters['trans']
        format_align = parameters['alignments']
        for a in format_align:
            k = tuple(a['key'])
            v = a['value']
            align[k] = v
    else:
        print("output alignment from model 1 parameters")
        translations = parameters
        inverse_tran = second_parameters

    all_alignments = []
    for sentence_index,pair in enumerate(corpus):
        if verbose and sentence_index % 500 == 0:
            print(sentence_index)
        if verbose and sentence_index == 38:
            write_to_file(all_alignments,outfile)
            print("output 37 lines")

        st = ""
        probability_french_for_sentence = []
        for french_index, word_french in enumerate(pair['fr'].split()):
            french_t = {z[0]:z[1] for z in second_parameters[word_french]}
            probability_french = []
            for en_index,word_english in enumerate(pair['en'].split()):
                probability_french.append(french_t[word_english])
            probability_per_word = ([ p/sum(probability_french) for p in probability_french])
            probability_french_for_sentence.append(probability_per_word)

        for en_index,word_english in enumerate(pair['en'].split()):
            max = 0
            save_j = 0
            if word_english in translations:
                tran = {z[0]:z[1] for z in translations[word_english]}
            else:
                continue

            for french_index,word_french in enumerate(pair['fr'].split()):
                french_t = probability_french_for_sentence[french_index]
                final_p = french_t[en_index] * tran[word_french]
                if final_p > max:
                    save_j = french_index
                    max = final_p

            if tran[None] < max:
                st += str(save_j)+"-"+str(en_index)+" "

        st = st[:-1]
        st += "\n"
        all_alignments.append(st)

    return all_alignments


def main():
    tran_file = tran_e_f
    parameters = get_corpus(tran_file)
    second_parameters = get_corpus(inverse_parameters)

    corpus = create_list_of_sentences(english_file,french_file, debug=False)
    all_strings = build_alignments(parameters,corpus,second_parameters)
    write_to_file(all_strings, outfile)


def write_to_file(all_strings, outfile):
    with open(outfile, 'w') as f:
        f.writelines(all_strings)


if __name__ == '__main__':
    import sys,os
    global outfile,verbose,english_file,french_file, tran_e_f,inverse_parameters
    english_file = sys.argv[1] if len(sys.argv) > 1 else "../data/hansards.e"
    french_file  = sys.argv[2] if len(sys.argv) > 2 else "../data/hansards.f"
    tran_e_f = sys.argv[3] if len(sys.argv) > 3 else "save_dict.json"
    inverse_parameters = sys.argv[4] if len(sys.argv) > 4 else "save_dict.json"
    outfile = sys.argv[5] if len(sys.argv) > 5 else "my_alignments.txt"
    verbose = sys.argv[6] if len(sys.argv) > 6 else "true"
    verbose = verbose.lower() == "true"
    main()

