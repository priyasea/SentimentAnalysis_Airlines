import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src.feature_engineering import FeatureEngineering


# -------------------------
# Load data
df = pd.read_csv("data/airline_tweets_train.csv")

sentiment_map = {
    "negative": 0,
    "positive": 1
}

df["sentiment"] = df["airline_sentiment"].map(sentiment_map)

# Drop original label column
df = df.drop("airline_sentiment", axis=1)
y = df["sentiment"]
X = df.drop("sentiment", axis=1)



# -------------------------
# Feature groups
# -------------------------
TEXT_COLUMN = "clean_text"

NUMERIC_FEATURES = [
    "text_length",
    "word_count",
    "neg_word_count",
    "all_caps_count",
    "exclamations",
    "has_negation",
    "retweet_count"
]


# -------------------------
# Column Transformer
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                max_df=0.9,
                min_df=5,
                stop_words="english"
            ),
            TEXT_COLUMN
        ),
        (
            "num",
            "passthrough",   
            NUMERIC_FEATURES
        )
    ],
    remainder="drop"
)


# -------------------------
# Full Pipeline
# -------------------------
pipeline = Pipeline(
    steps=[
        ("feature_engineering", FeatureEngineering()),
        ("preprocessing", preprocessor),
        ("classifier", LinearSVC(
                        C=0.1,
                        class_weight="balanced",
                        random_state=42
                            ))
    ]
)




print("\nTraining model on full dataset...")
pipeline.fit(X, y)
print(" Model trained successfully!")


# -------------------------
# Save pipeline
# -------------------------
model_path = "models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

