import re
import html
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer that:

    - Cleans raw tweet text
    - Creates handcrafted numeric text features
    - Drops unused identifier columns
    - Produces clean_text for downstream TF-IDF
    """

    def __init__(self):
        self.negative_words = [
            "bad", "worst", "waste", "junk", "disappointed", "hate",
            "cheap", "difficult", "hungry", "fault", "cancelled",
            "delayed", "stuck", "suck", "ridiculous", "lost", "terrible"
        ]

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _remove_hashtags_mentions(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'@\w+|#\w+', '', text)
        text = html.unescape(text)
        return text

    @staticmethod
    def _clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r"[^a-z\s!?]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # -------------------------
    # Sklearn API
    # -------------------------
    def fit(self, X, y=None):
        """
        No fitting required.
        Exists only to satisfy sklearn API.
        """
        return self

    def transform(self, X):
        X = X.copy()

        # -------------------------
        # Text cleanup
        # -------------------------
        X["text"] = X["text"].apply(self._remove_hashtags_mentions)

        # -------------------------
        # Feature engineering
        # -------------------------
        X["text_length"] = X["text"].str.len()
        X["word_count"] = X["text"].str.split().str.len()

        X["neg_word_count"] = (
            X["text"]
            .str.lower()
            .str.count("|".join(self.negative_words))
        )

        X["all_caps_count"] = (
            X["text"]
            .str.findall(r"\b[A-Z]{2,}\b")
            .str.len()
        )

        X["exclamations"] = X["text"].apply(lambda x: x.count("!"))

        X["has_negation"] = (
            X["text"]
            .str.lower()
            .str.contains(
                r"\b(?:not|no|never|don\'t|doesn\'t|didn\'t|can\'t|won\'t)\b",
        regex=True
            )
            .astype(int)
        )

        # -------------------------
        # Clean text for TF-IDF
        # -------------------------
        X["clean_text"] = X["text"].apply(self._clean_text)

        # -------------------------
        # Drop unused columns
        # -------------------------
        X = X.drop(
            ["Id", "user_timezone", "airline"],
            axis=1,
            errors="ignore"
        )

        return X
