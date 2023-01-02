import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]
    
    
class GetMunicipioFeatures(BaseEstimator, TransformerMixin):
    

    
    def __init__(self):
        self.piores_municipios = ["Macapá", "Porto Velho", 
                                  "Ananindeua", "Belém", 
                                  "Santarém", "São Gonçalo", 
                                  "Duque de Caxias", "Rio Branco", 
                                  "Belford Roxo"]

        self.best_municipios = ["Santos", "Maringá", 
                                "Uberlândia", "Franca", 
                                "Limeira", "Piracicaba", 
                                "Cascavel", "São Paulo",
                                "São José do Rio Preto", "Suzano", 
                                "São Caetano do Sul"]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        worse_municpios = X.isin(self.piores_municipios).astype(np.int8)
        best_municpios = X.isin(self.best_municipios).astype(np.int8)
        worse_municpios.columns = [f"WORSE_{column}" for column in worse_municpios.columns]
        best_municpios.columns = [f"BEST_{column}" for column in best_municpios.columns]

        return pd.concat([worse_municpios, best_municpios], axis = 1)
