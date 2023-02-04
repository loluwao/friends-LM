from curses.ascii import isalpha

import nltk
from nltk import word_tokenize
from soupsieve.util import lower
from collections import Counter


#nltk.download("punkt")
#from nltk.tokenize import word_tokenize
'''
moby = open('mobydick(1).txt', "r", encoding='utf-8')
contents = moby.read()
moby.close()

#print(len(word_tokenize("Krishna got their Ph.D. in computer science after they built a state-of-the-art model for NER (and wrote a long dissertation")))
#contents = contents.lower()
vocab = word_tokenize(contents)
#print("heyyy " + str(len(Counter(vocab))))
#print(len(vocab))
print(vocab[:100])
unique_vocab = []
for token in vocab:
    #if token.isalpha():
    if not token in unique_vocab:
        unique_vocab.append(token)

print(len(unique_vocab))
print(unique_vocab[:50])
'''

def levenshtein(source, target):
    """
    This function calculates the levenshtein distance (minimum edit distance w/ cost 1 for operations)
    between a source word and a target word.

    Parameters:
        source - string word 1
        target - string word 2
    Return:
        int final distance between the two words
    """
    n = len(source)
    m = len(target)

    matrix = [[0 for i in range(m + 1)] for i in range(n + 1)]  # create matrix

    matrix =[]
    for i in range(n + 1):
        row = []
        for j in range(m + 1):
            row.append(0)
        matrix.append(row)
    #print(matrix)
    # point 1

    for row_index in range(1, n + 1):
        matrix[row_index][0] = matrix[row_index - 1][0] + 1  # levenshtein has delete cost 1

    for col_index in range(1, m + 1):
        matrix[0][col_index] = matrix[0][col_index - 1] + 1  # levenshtein has insert cost 1

    print(matrix)
    # point 2

    for row_index in range(1, n + 1):
        for col_index in range(1, m + 1):
            # since our strings are 0 indexed but our matrix is 1-indexed
            # use substitute cost 2
            sub_cost = 0 if source[row_index - 1] == target[col_index - 1] else 2

            # use insert and delete cost 1
            matrix[row_index][col_index] = min(matrix[row_index - 1][col_index] + 1, \
                                               matrix[row_index - 1][col_index - 1] + sub_cost, \
                                               matrix[row_index][col_index - 1] + 1)
            # point 3

    #print(matrix)
    return matrix[n][m]

levenshtein("cat", "bats")