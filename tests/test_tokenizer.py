from papersearch.tokenizer import stem, terms, words


def test_words_are_lowercase_and_split_on_punctuation():
    assert words("State-of-the-art GNNs, in 2024!") == ["state", "of", "the", "art", "gnns", "in", "2024"]


def test_terms_drop_stopwords_and_single_characters():
    assert terms("The model is a kind of network") == ["model", "kind", "network"]
    assert terms("k means and C code") == ["mean", "code"]


def test_plurals_become_singular():
    assert stem("networks") == "network"
    assert stem("queries") == "query"
    assert stem("classes") == "class"
    assert stem("approaches") == "approach"
    assert stem("boxes") == "box"


def test_words_that_only_look_plural_are_kept():
    for word in ("bias", "analysis", "corpus", "class", "lens", "gas"):
        assert stem(word) == word


def test_irregular_plurals():
    assert stem("matrices") == "matrix"
    assert stem("biases") == "bias"


def test_singular_and_plural_give_the_same_term():
    assert terms("neural network") == terms("Neural Networks")
