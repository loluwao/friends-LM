# imports go here
import sys
import string
from collections import Counter
import random

# feel free to add imports.

"""
Don't forget to put your name and a file comment here.

This is an individual homework.
"""

# Feel free to implement more helper functions


"""
Provided helper functions
"""


def read_sentences(filepath):
    """
  Reads contents of a file line by line.
  Parameters:
    filepath (str): file to read from
  Return:
    list of strings
  """
    f = open(filepath, "r")
    sentences = f.readlines()
    f.close()
    return sentences


def create_ngrams(n, sentences, end):
    gram = ()
    grams = []
    for sentence in sentences:
        sentence = sentence.split()
        for i in range(len(sentence) - n + 1):
            for j in range(n):
                gram += (sentence[i + j],)
            grams.append(gram)
            gram = ()

    return grams


def tokenize(sentences):
    tokens = []
    for sentence in sentences:
        sublist = sentence.split()
        for word in sublist:
            # if word not in tokens:
            tokens.append(word)
    return tokens


def get_data_by_character(filepath):
    """
  Reads contents of a script file line by line and sorts into 
  buckets based on speaker name.
  Parameters:
    filepath (str): file to read from
  Return:
    dict of strings to list of strings, the dialogue that speaker speaks
  """
    char_data = {}
    script_file = open(filepath, "r", encoding="utf-8")
    for line in script_file:
        # extract the part between <speaker> tags
        speakers = line[line.index("<speakers>") + len("<speakers>"): line.index("</speakers>")].strip()
        if not speakers in char_data:
            char_data[speakers] = []
        char_data[speakers].append(line)
    return char_data


"""
This is your Language Model class
"""


class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"

    def __init__(self, n_gram, is_laplace_smoothing, line_begin="<line>", line_end="</line>"):
        """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether or not to use Laplace smoothing
      line_begin (str): the token designating the beginning of a line
      line_end (str): the token designating the end of a line
    """

        self.line_begin = line_begin
        self.line_end = line_end
        # your other code here
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.all_grams = {}
        self.all_counts = {}
        self.sentences = []
        self.tokens = []
        self.vocab = []
        self.vocab_size = 0

    def train(self, sentences):
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with line_begin and end with line_end
    Parameters:
      sentences (list): list of strings, one string per line in the training file

    Returns:
    None
    """
        unknowns = []

        # initialize tokens
        self.tokens = tokenize(sentences)
        self.vocab_size = len(self.tokens)
        cc = Counter(self.tokens)

        # set # of <UNK> to zero
        ccc = cc.copy()
        ccc[self.UNK] = 0

        # count unknown tokens
        for tok in cc:
            if cc[tok] == 1:
                ccc[self.UNK] += 1
                ccc.pop(tok)
        if ccc[self.UNK] == 0:
            del ccc[self.UNK]

        self.tokens = list(ccc.keys()).copy()
        #print("done tokenizing")

        # initialize vocab
        for i in range(len(sentences)):
            sentence = sentences[i]
            toks = sentence.split()
            new_toks = []
            for tok in toks:
                t = tok

                if tok not in self.tokens:
                    t = self.UNK
                new_toks.append(t)
            self.sentences.append(" ".join(new_toks))

        # create all n_grams (dictionary of dictionaries) and counts for n-grams
        self.all_grams[self.n_gram] = create_ngrams(self.n_gram, self.sentences, self.line_end)
        self.all_counts[self.n_gram] = Counter(list(self.all_grams[self.n_gram]))

        # if n > 1, create n-1 grams and counts for n-1 grams
        if self.n_gram > 1:
            self.all_grams[self.n_gram - 1] = create_ngrams(self.n_gram - 1, self.sentences, self.line_end)
            self.all_counts[self.n_gram - 1] = Counter(list(self.all_grams[self.n_gram - 1]))

        #print("done counting n-grams and n-1-grams")


    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of

    Returns:
      float: the probability value of the given string for this model
    """

        toks = sentence.split()
        new_toks = []
        for tok in toks:
            t = tok
            if tok not in self.tokens:
                t = self.UNK
            new_toks.append(t)

        sentence_grams = create_ngrams(self.n_gram, [" ".join(new_toks)], self.line_end)

        probs = []
        for gram in sentence_grams:
            num = self.all_counts[self.n_gram][gram]
            denom = 0
            if self.n_gram > 1:
                temp = list(gram)
                temp = temp[:-1]
                temp = tuple(temp)
                denom = self.all_counts[self.n_gram - 1][temp]
            else:
                for g in self.all_counts[self.n_gram]:
                    denom = self.vocab_size
            if self.is_laplace_smoothing:
                num += 1
                denom += len(self.tokens)

            if denom == 0:
                probs.append(0)
            else:
                probs.append(num / denom)
        quotient = 1
        for x in probs:
            quotient = quotient * x
        return quotient

    def highest_prob(self, count_dict):
        #random.shuffle(count_dict) # works

        # my shuffle
        keys = list(count_dict.keys())
        random.shuffle(keys)
        new_dict = {}
        for key in keys:
            new_dict[key] = count_dict[key]
        count_dict = new_dict.copy()
        total = 0

        # count total occurences of (w1, wn) (works)
        for count in count_dict.values():
            total += count

        # calculate probability distribution of each n-gram ( works )
        probs = {}
        for gram, count in count_dict.items():
            probs[gram] = count_dict[gram] / total

        gram_w_max = ()
        max = 0
        for gram, count in probs.items():
            if count > max:
                gram_w_max = gram

        return gram_w_max


    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

    Returns:
      str: the generated sentence
    """
        print("generating sentence...")
        sent_arr = [self.line_begin]
        curr = {}
        if self.n_gram > 1:
            while sent_arr[-1] != self.line_end:

                for gram, count in self.all_counts[self.n_gram].items():
                     if gram[0] == sent_arr[-1]:
                        curr[gram] = count
                            #print(gram, count)
                    #print("counts of n-grams starting w " + sent_arr[-1] + ": " + str(curr))

                    # find next gram w highest prob
                nxt = list(self.highest_prob(curr))
                sent_arr += nxt[1:]
        else:
            total = 0
            counts = self.all_counts[1]
            del counts[(self.line_begin,)]

            for val in counts.values():
                total += val
            for gram, count in counts.items():
                curr[gram] = count / total

            done = False
            while not done:
                max_prob = max(curr.values())
                gram_w_max = ""
                for gram in curr.keys():
                    if curr[gram] == max_prob:
                        gram_w_max = gram[0]
                        del curr[gram]
                        break

                sent_arr.append(gram_w_max)
                if sent_arr[-1] == self.line_end:
                    done = True

        sentence = " ".join(sent_arr)
        print("done generating sentence")
        return sentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate

    Returns:
      list: a list containing strings, one per generated sentence
    """
        return [self.generate_sentence() for i in range(n)]

def main():
    # TODO: implement the rest of this!
    ngram = int(sys.argv[1])
    training_path = sys.argv[2]
    testing_path = sys.argv[3]
    line_begin = sys.argv[4]
    if len(sys.argv) == 5:
        print("Runnning for", ngram, "model")

        # instantiate a language model like....
        ngram_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
        ngram_lm.train(read_sentences(training_path))
        sentences = ngram_lm.generate(1)
        for sentence in sentences:
            print(sentence)
    else:
        # code where you compare the different characters

        # create models
        print("creating each character's language model...")
        monica_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
        ross_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
        joey_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
        phoebe_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
        rachel_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
        chandler_lm = LanguageModel(ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")

        # create character specific data
        print("training each model on its character's dialogue...\n")
        monica_dialogue = [sentence.replace("<speakers> Monica Geller </speakers>", "") for sentence in
                           read_sentences(training_path) if "<speakers> Monica Geller" in sentence]
        ross_dialogue = [sentence.replace("<speakers> Ross Geller </speakers>", "") for sentence in
                         read_sentences(training_path) if "<speakers> Ross Geller" in sentence]
        joey_dialogue = [sentence.replace("<speakers> Joey Tribbiani </speakers>", "") for sentence in
                         read_sentences(training_path) if "<speakers> Joey Tribbiani" in sentence]
        phoebe_dialogue = [sentence.replace("<speakers> Phoebe Buffay </speakers>", "") for sentence in
                           read_sentences(training_path) if "<speakers> Phoebe Buffay" in sentence]
        rachel_dialogue = [sentence.replace("<speakers> Rachel Green </speakers>", "") for sentence in
                           read_sentences(training_path) if "<speakers> Rachel Green" in sentence]
        chandler_dialogue = [sentence.replace("<speakers> Chandler Bing </speakers>", "") for sentence in
                             read_sentences(training_path) if "<speakers> Chandler Bing" in sentence]

        # train models
        monica_lm.train(monica_dialogue)
        ross_lm.train(ross_dialogue)
        joey_lm.train(joey_dialogue)
        phoebe_lm.train(phoebe_dialogue)
        rachel_lm.train(rachel_dialogue)
        chandler_lm.train(chandler_dialogue)


        # monica
        print("calculating who's most similar to Monica...")
        monica_similar = {}
        num = 0
        for sentence in ross_dialogue:
            num += monica_lm.score(sentence)
        monica_similar["Ross"] = num / len(ross_dialogue)
        num = 0
        for sentence in joey_dialogue:
            num += monica_lm.score(sentence)
        monica_similar["Joey"] = num / len(joey_dialogue)
        num = 0
        for sentence in phoebe_dialogue:
            num += monica_lm.score(sentence)
        monica_similar["Phoebe"] = num / len(phoebe_dialogue)
        num = 0
        for sentence in rachel_dialogue:
            num += monica_lm.score(sentence)
        monica_similar["Rachel"] = num / len(rachel_dialogue)
        num = 0
        for sentence in chandler_dialogue:
            num += monica_lm.score(sentence)
        monica_similar["Chandler"] = num / len(chandler_dialogue)

        most_similar = list(monica_similar.keys())[list(monica_similar.values()).index(max(monica_similar.values()))]

        print("{} is most similar to Monica, receiving an average score of {}.".format(most_similar, max(monica_similar.values())))

        # ross
        print("calculating who's most similar to Ross...")
        ross_similar = {}
        num = 0
        for sentence in monica_dialogue:
            num += ross_lm.score(sentence)
        ross_similar["Monica"] = num / len(monica_dialogue)
        num = 0
        for sentence in joey_dialogue:
            num += ross_lm.score(sentence)
        ross_similar["Joey"] = num / len(joey_dialogue)
        num = 0
        for sentence in phoebe_dialogue:
            num += ross_lm.score(sentence)
        ross_similar["Phoebe"] = num / len(phoebe_dialogue)
        num = 0
        for sentence in rachel_dialogue:
            num += ross_lm.score(sentence)
        ross_similar["Rachel"] = num / len(rachel_dialogue)
        num = 0
        for sentence in chandler_dialogue:
            num += ross_lm.score(sentence)
        ross_similar["Chandler"] = num / len(chandler_dialogue)

        most_similar = list(ross_similar.keys())[list(ross_similar.values()).index(max(ross_similar.values()))]

        print("{} is most similar to Ross, receiving an average score of {}.".format(most_similar,
                                                                                       max(ross_similar.values())))

        # joey
        print("calculating who's most similar to Joey...")
        joey_similar = {}
        num = 0
        for sentence in monica_dialogue:
            num += joey_lm.score(sentence)
        joey_similar["Monica"] = num / len(monica_dialogue)
        num = 0
        for sentence in ross_dialogue:
            num += joey_lm.score(sentence)
        joey_similar["Ross"] = num / len(ross_dialogue)
        num = 0
        for sentence in phoebe_dialogue:
            num += joey_lm.score(sentence)
        joey_similar["Phoebe"] = num / len(phoebe_dialogue)
        num = 0
        for sentence in rachel_dialogue:
            num += joey_lm.score(sentence)
        joey_similar["Rachel"] = num / len(rachel_dialogue)
        num = 0
        for sentence in chandler_dialogue:
            num += joey_lm.score(sentence)
        joey_similar["Chandler"] = num / len(chandler_dialogue)

        most_similar = list(joey_similar.keys())[list(joey_similar.values()).index(max(joey_similar.values()))]

        print("{} is most similar to Joey, receiving an average score of {}.".format(most_similar,
                                                                                       max(joey_similar.values())))

        # phoebe
        print("calculating who's most similar to Phoebe...")
        phoebe_similar = {}
        num = 0
        for sentence in monica_dialogue:
            num += phoebe_lm.score(sentence)
        phoebe_similar["Monica"] = num / len(monica_dialogue)
        num = 0
        for sentence in ross_dialogue:
            num += phoebe_lm.score(sentence)
        phoebe_similar["Ross"] = num / len(ross_dialogue)
        num = 0
        for sentence in joey_dialogue:
            num += phoebe_lm.score(sentence)
        phoebe_similar["Joey"] = num / len(joey_dialogue)
        num = 0
        for sentence in rachel_dialogue:
            num += phoebe_lm.score(sentence)
        phoebe_similar["Rachel"] = num / len(rachel_dialogue)
        num = 0
        for sentence in chandler_dialogue:
            num += phoebe_lm.score(sentence)
        phoebe_similar["Chandler"] = num / len(chandler_dialogue)

        most_similar = list(phoebe_similar.keys())[list(phoebe_similar.values()).index(max(phoebe_similar.values()))]

        print("{} is most similar to Phoebe, receiving an average score of {}.".format(most_similar,
                                                                                       max(phoebe_similar.values())))

        # rachel
        print("calculating who's most similar to Rachel...")
        rachel_similar = {}
        num = 0
        for sentence in monica_dialogue:
            num += rachel_lm.score(sentence)
        rachel_similar["Monica"] = num / len(monica_dialogue)
        num = 0
        for sentence in ross_dialogue:
            num += rachel_lm.score(sentence)
        rachel_similar["Ross"] = num / len(ross_dialogue)
        num = 0
        for sentence in joey_dialogue:
            num += rachel_lm.score(sentence)
        rachel_similar["Joey"] = num / len(joey_dialogue)
        num = 0
        for sentence in phoebe_dialogue:
            num += rachel_lm.score(sentence)
        rachel_similar["Phoebe"] = num / len(phoebe_dialogue)
        num = 0
        for sentence in chandler_dialogue:
            num += rachel_lm.score(sentence)
        rachel_similar["Chandler"] = num / len(chandler_dialogue)

        most_similar = list(rachel_similar.keys())[list(rachel_similar.values()).index(max(rachel_similar.values()))]

        print("{} is most similar to Rachel, receiving an average score of {}.".format(most_similar,
                                                                                       max(rachel_similar.values())))

        # chandler
        print("calculating who's most similar to Chandler...")
        chandler_similar = {}
        num = 0
        for sentence in monica_dialogue:
            num += chandler_lm.score(sentence)
        chandler_similar["Monica"] = num / len(monica_dialogue)
        num = 0
        for sentence in ross_dialogue:
            num += chandler_lm.score(sentence)
        chandler_similar["Ross"] = num / len(ross_dialogue)
        num = 0
        for sentence in joey_dialogue:
            num += chandler_lm.score(sentence)
        chandler_similar["Joey"] = num / len(joey_dialogue)
        num = 0
        for sentence in phoebe_dialogue:
            num += chandler_lm.score(sentence)
        chandler_similar["Phoebe"] = num / len(phoebe_dialogue)
        num = 0
        for sentence in rachel_dialogue:
            num += chandler_lm.score(sentence)
        chandler_similar["Rachel"] = num / len(rachel_dialogue)

        print(chandler_similar)
        most_similar = list(chandler_similar.keys())[list(chandler_similar.values()).index(max(chandler_similar.values()))]

        print("{} is most similar to Chandler, receiving an average score of {}.".format(most_similar,
                                                                                       max(chandler_similar.values())))
if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) < 5:
        print("Usage:", "python lm.py ngram training_file.txt testingfile.txt line_begin [character]")
        sys.exit(1)

    main()