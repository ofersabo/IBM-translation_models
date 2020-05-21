import json
import pprint
import sys
from operator import itemgetter

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


def get_corpus(filename):
    '''
    Load corpus file located at `filename` into a list of dicts
    '''
    with open(filename, 'r') as f:
        corpus = json.load(f)

    # if VERBOSE:
    #     print(corpus, file=sys.stderr)
    return corpus

def build_alignments(all_translations, corpus):
    all_alignments = []
    for sentence_index,pair in enumerate(corpus):
        if sentence_index % 500 == 0:
            print(sentence_index)
        st = ""
        for i,word_english in enumerate(pair['en'].split()):
            max = 0
            save_j = 0
            if word_english in all_translations:
                tran = {i[0]:i[1] for i in all_translations[word_english]}
            else:
                save_j = 0
                st += str(save_j)+"-"+str(i)+" "
                continue

            for j,word_french in enumerate(pair['fr'].split()):
                if word_french in tran and tran[word_french] > max:
                    save_j = j
                    max = tran[word_french]

            st += str(save_j)+"-"+str(i)+" "
        st = st[:-1]
        st += "\n"
        all_alignments.append(st)

    return all_alignments


def main(tran_file,outfile):
    translations = get_corpus(tran_file)
    corpus = create_list_of_sentences("../data/hansards.e", "../data/hansards.f", debug=False)
    all_strings = build_alignments(translations,corpus)
    if outfile:
        with open(outfile, 'w') as f:
            f.writelines(all_strings)
    else:
        for l in all_strings:
            print(l)


if __name__ == '__main__':
    import sys,os
    x = os.getcwd()
    print(x)
    tran = sys.argv[1] if len(sys.argv) > 1 else "save_dict.json"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "my_alignments.txt"
    main(tran_file = tran,outfile=outfile)

