import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, all_genres):
        self.mlb = MultiLabelBinarizer(classes=all_genres)
        self.fitted = False  # Track if the MultiLabelBinarizer has been fitted

    def fit(self, X, y=None):
        # No need to fit the MultiLabelBinarizer
        self.fitted = True  # Set fitted to True after initialization
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("The MultiLabelBinarizer must be fitted before transforming data.")
        
        X.dropna(subset=['genre'], inplace=True)
        X.reset_index(drop=True, inplace=True)
        
        X['genre'] = X['genre'].str.strip()
        
        # Transform the genre column
        genres = self.mlb.transform(X['genre'].str.split(', '))
        genre_matrix = pd.DataFrame(genres, columns=self.mlb.classes_, index=X.index)
        
        X = pd.concat([X, genre_matrix], axis=1)
        
        # No need to limit the columns
        # last_genre_column_index = 49
        # X = X.iloc[:, :last_genre_column_index + 1]
        
        return X