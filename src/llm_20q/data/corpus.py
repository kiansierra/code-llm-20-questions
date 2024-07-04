import itertools
import string

import pandas as pd

from .keywords_v1 import KEYWORDS_JSON_V1
from .keywords_v2 import KEYWORDS_JSON_V2

KEYWORDS = eval(KEYWORDS_JSON_V1) + eval(KEYWORDS_JSON_V2)
CATEGORIES = list({elem["category"] for elem in KEYWORDS})
ALL_KEYWORDS = list(
    itertools.chain(*[itertools.chain(item["keyword"] for item in elem["words"]) for elem in KEYWORDS])
)
CATEGORY_BASE_QUESTIONS = "Is the Keyword a {category}? {answer}"
LETTER_BASE_QUESTIONS = "Does the Keyword start with {letter}? {answer}"


def build_corpus() -> pd.DataFrame:
    """
    Builds a corpus dataframe from a list of keywords and their associated categories and alternatives.

    Returns:
        pd.DataFrame: The corpus dataframe containing the keywords, categories, and alternatives.
    """
    dfs = []
    for elem in KEYWORDS:
        df = pd.DataFrame(elem["words"])
        df["category"] = elem["category"]
        dfs.append(df)
    corpus_df = pd.concat(dfs)
    corpus_df = corpus_df.explode("alts")
    corpus_df["alts"] = corpus_df["alts"].fillna("")
    corpus_df = corpus_df.groupby("keyword", as_index=False).agg(
        {"category": lambda x: x.unique(), "alts": lambda x: x.unique()}
    )
    corpus_df = corpus_df.explode("alts")
    corpus_df["letter"] = corpus_df["keyword"].apply(lambda x: x[0])
    return corpus_df


def build_category_questions() -> pd.DataFrame:
    """
    Builds a DataFrame containing category-based questions.

    Returns:
        pd.DataFrame: A DataFrame containing category-based questions.
    """

    category_questions_positive = [
        {
            "query_category": category,
            "question": CATEGORY_BASE_QUESTIONS.format(category=category, answer="yes"),
            "score": 1,
            "answer": "yes",
        }
        for category in CATEGORIES
    ]
    category_questions_negative = [
        {
            "query_category": category,
            "question": CATEGORY_BASE_QUESTIONS.format(category=category, answer="no"),
            "score": -1,
            "answer": "no",
        }
        for category in CATEGORIES
    ]
    category_queries = pd.DataFrame(category_questions_positive + category_questions_negative)
    return category_queries


def build_letter_based_questions() -> pd.DataFrame:
    """
    Builds a DataFrame containing letter-based questions.

    Returns:
        pd.DataFrame: A DataFrame containing letter-based questions.
    """

    letter_questions_positive = [
        {
            "query_letter": letter,
            "question": LETTER_BASE_QUESTIONS.format(letter=letter, answer="yes"),
            "score": 1,
            "answer": "yes",
        }
        for letter in string.ascii_lowercase
    ]
    letter_questions_negative = [
        {
            "query_letter": letter,
            "question": LETTER_BASE_QUESTIONS.format(letter=letter, answer="no"),
            "score": -1,
            "answer": "no",
        }
        for letter in string.ascii_lowercase
    ]
    letter_queries = pd.DataFrame(letter_questions_positive + letter_questions_negative)
    return letter_queries
