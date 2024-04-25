import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.fitted = False  # Track if the MultiLabelBinarizer has been fitted

    def fit(self, X, y=None):
        self.mlb.fit(X['genre'].str.split(', '))
        self.fitted = True  # Set fitted to True after fitting
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("The MultiLabelBinarizer must be fitted before transforming data.")
        
        X.dropna(subset=['genre'], inplace=True)
        X.reset_index(drop=True, inplace=True)
        
        X['genre'] = X['genre'].str.strip()
        
        genres = self.mlb.transform(X['genre'].str.split(', '))
        genre_matrix = pd.DataFrame(genres, columns=self.mlb.classes_, index=X.index)
        
        X = pd.concat([X, genre_matrix], axis=1)
        
        last_genre_column_index = 49
        X = X.iloc[:, :last_genre_column_index + 1]
        
        return X
































#import pandas as pd
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.preprocessing import MultiLabelBinarizer

#class Preprocessor(BaseEstimator, TransformerMixin):
#    def __init__(self):
#        self.mlb = MultiLabelBinarizer()

#    def fit(self, X, y=None):
#        self.mlb.fit(X['genre'].str.split(', '))
#        return self

#    def transform(self, X):
#        X.dropna(subset=['genre'], inplace=True)
#        X.reset_index(drop=True, inplace=True)
        
#        X['genre'] = X['genre'].str.strip()
        
#        genres = self.mlb.transform(X['genre'].str.split(', '))
#        genre_matrix = pd.DataFrame(genres, columns=self.mlb.classes_, index=X.index)
        
#        X = pd.concat([X, genre_matrix], axis=1)
        
#        last_genre_column_index = 49
#        X = X.iloc[:, :last_genre_column_index + 1]
        
#        return X