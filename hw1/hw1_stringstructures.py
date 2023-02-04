# An assortment of problems to learn and review python
# and string manipulation as well as some of our favorite
# python data structures--lists, dicts, sets, tuples
# (and a small amount of graphing)
# implement these functions according to the provided doc strings
import string
from collections import Counter
import matplotlib.pyplot as plt
import operator

ls = ["hi", "yo", "hello", "HI", "hi", "hi", "hi"]
text = "HIIIII HEYYYYYYY whAT's up"

# Stand alone functions
def switch_case(text: str) -> str:
    """
    This functions switches the case of every individual
    character in a string.
    Parameters:
    text - str - string to be switched
    Return:
    str - switched version of the input
    """
    return text.swapcase()


def bad_tokenize(text: str) -> list:
    """
    Tokenize a string by splitting words on all whitespace.
    Separate all punctuation from the characters that it is adjacent to.
    Parameters:
    text - str - string to be tokenized
    Return:
    list - list of individual tokens
    """
    ls = []
    word = ""
    for c in text:
        if c.isalpha() or c.isnumeric():
            word += c
        elif c in string.punctuation:
            if len(word) > 0:
                ls.append(word)
            word = ""
            ls.append(c)
        elif c.isspace():
            if len(word) > 0:
                ls.append(word)
            word = ""
    if len(word) > 0:
        ls.append(word)
    return ls

def case_fold(words: list):
    """
    Convert every word in a list of words to lower case.
    Parameters:
    words - list - list of words to be converted to lower case
    Return:
    either None or list, depending on your implementation
    """
    return [word.lower() for word in words]


def count_dict(words: list) -> dict:
    """
    Create a dictionary that maps individual words (strings)
    to their counts.
    Do not use a Counter.
    Parameters:
    words - list - list of words to be counted
    Return:
    dict - mapping strings to ints
    """
    words = case_fold(words)
    return {word : operator.countOf(words, word) for (word) in words}


def count_counter(words: list) -> Counter:
    """
    Create a Counter that maps individual words (strings)
    to their counts.
    Use a Counter.
    Parameters:
    words - list - list of words to be counted
    Return:
    Counter - mapping strings to ints
    """
    words = case_fold(words)
    return Counter(words)


def get_vocab_size(words: list) -> int:
    """
    Return the number of unique words contained in this list
    of words.
    Parameters:
    words - list - list of words
    Return:
    int size of vocabulary for these words
    """
    words = case_fold(words)
    return len(Counter(words))


def heaps_law(tokens: list) -> None:
    """
    Take the first 1%, then 2%, then 3%... up to 100% of
    the given tokens.
    Create a graph of number of tokens (on the x-axis) vs. size of vocabulary (on
the y-axis)

    plt.plot(x_coords, y_coords)
    plt.xlabel("x label string")
    plt.ylabel("y label string")
    plt.title("title string")
    plt.savefig("heapslaw.pdf", bbox_inches="tight")
    plt.show()
    Parameters:
    tokens - list - list of tokens to be analyzed
    Return:
    None
    """
    x = [i * get_vocab_size(tokens) / 100 for i in range(100)]
    y = [10 * x[i] ** 0.4 for i in range(100)]

    plt.plot(x, y)
    plt.xlabel("Number of tokens (N)")
    plt.ylabel("Size of vocab |V|")
    plt.savefig("heapslaw.pdf", bbox_inches="tight")
    plt.show()


def main():
    print("Running standalone functions")
    print("original text: " + text)
    print("switch_case: " + switch_case(text))
    print("bad_tokenize: " + str(bad_tokenize(text)))

    print("original word list: " + str(ls))
    print("case_fold: " +  str(case_fold(ls)))
    print("count_dict: " + str(count_dict(ls)))
    print("count_counter: " + str(count_counter(ls)))
    print("get_vocab_size: " + str(get_vocab_size(ls)))

    # TODO: load in tokens from a text file of your choice. Load in a minimum of
    # 10000 tokens
    txt = open('world192.txt', "r", encoding='utf-8')
    contents = txt.read()
    txt.close()
    contents = contents.replace("\n", " ")
    words = bad_tokenize(contents)

    counts = count_counter(words)

    # TODO: call each standalone function on some
    # example data
    heaps_law(list(counts.keys()))

main()