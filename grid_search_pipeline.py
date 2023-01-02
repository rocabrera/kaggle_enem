import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from utils import clean_target
from categorical_ordinal import get_categorical_ordinal_columns
from categorical_nominal import get_categorical_nominal_columns
from columns_transformers import ColumnSelector, GetMunicipioFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder, MinMaxScaler, PolynomialFeatures)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

train_df = pd.read_parquet("data/train.parquet")
clean_target(train_df)
#test= pd.read_parquet("data/test.parquet")

categorical_ordinal_columns = get_categorical_ordinal_columns(train_df)
qtd_categorical_ordinal_columns=len(categorical_ordinal_columns)
print(f"Number of categorial ordinal features: {qtd_categorical_ordinal_columns}")

categorical_nominal_columns = get_categorical_nominal_columns(train_df)
qtd_categorical_nominal_columns = len(categorical_nominal_columns)
print(f"Number of categorial nominal features: {qtd_categorical_nominal_columns}")

drop_columns = ["NU_INSCRICAO", 
                "CO_ESCOLA", 
                "NO_MUNICIPIO_ESC", 
                "SG_UF_ESC", 
                "TP_DEPENDENCIA_ADM_ESC", 
                "TP_LOCALIZACAO_ESC", 
                "TP_SIT_FUNC_ESC"]
qtd_drop_columns = len(drop_columns)
print(f"Number of columns dropped: {qtd_drop_columns}")

custom_features = ["NO_MUNICIPIO_RESIDENCIA", "NO_MUNICIPIO_NASCIMENTO", "NO_MUNICIPIO_PROVA"]

target_columns = train_df.filter(regex="NU_NOTA").columns.tolist()
qtd_target_columns = len(target_columns)
print(f"Number of targets: {qtd_target_columns}")

numerical_columns = ["NU_IDADE"]
qtd_numerical_columns = len(numerical_columns)
print(f"Number of targets: {qtd_numerical_columns}")

target_columns = train_df.filter(regex="NU_NOTA").columns.tolist()
qtd_target_columns = len(target_columns)
print(f"Number of targets: {qtd_target_columns}")

all_columns = drop_columns + categorical_nominal_columns + categorical_ordinal_columns + numerical_columns + target_columns
qtd_total = qtd_drop_columns + qtd_categorical_nominal_columns + qtd_categorical_ordinal_columns + qtd_numerical_columns + qtd_target_columns
print(f"Total columns: {qtd_total}")

"""
Variáveis categóricas com dados ordinais que tem dados faltantes:
- TP_ENSINO: Suposto que NaN representa a categoria faltante descrita nos metadados.
- TP_STATUS_REDACAO: Mapeado para outra classe (Faltou na prova)
"""
categorical_ordinal_pipe = Pipeline([
    ('selector', ColumnSelector(categorical_ordinal_columns)),
    ('imputer', SimpleImputer(missing_values=np.nan, 
                              strategy='constant', 
                              fill_value=0)),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('extra_features', PolynomialFeatures(degree=2))
])

"""
Variáveis categóricas com dados ordinais que tem dados faltantes:
- SG_UF_NASCIMENTO: Mapeado para uma nova categoria
"""
categorical_nominal_pipe = Pipeline([
    ('selector', ColumnSelector(categorical_nominal_columns)),
    ('imputer', SimpleImputer(missing_values=np.nan, 
                              strategy='constant', 
                              fill_value="missing")),
    ('encoder', OneHotEncoder(drop="first", handle_unknown='ignore'))
])

numerical_pipe = Pipeline([
    ('selector', ColumnSelector(numerical_columns)),
    ('imputer', SimpleImputer(missing_values=np.nan, 
                              strategy='constant', 
                              fill_value=0)),
    ('scaler', MinMaxScaler())
])

extra_features_pipe = Pipeline([
    ('selector', ColumnSelector(custom_features)),
    ('extra_features', GetMunicipioFeatures())
])

preprocessor = FeatureUnion([
    ('categorical_ordinal', categorical_ordinal_pipe),
    ('categorical_nominal', categorical_nominal_pipe),
    ('numerical', numerical_pipe),
    ('municipio_features', extra_features_pipe)
])

param_grid = {"model__n_estimators" : [50, 100],
               "model__max_features" : [10, 30, 50]},

kwargs_regressor = {"n_jobs":-1,
                    "verbose":2}

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(**kwargs_regressor))
])

def presence_filter(df, key):
    cond = df.filter(regex=f"PRESENCA_{key}|STATUS_{key}").iloc[:,0] == 1
    return df.loc[cond, :], df.loc[~cond, :]


y_structure = {"CN":[], 
               "CH":[],
               "LC":[],
               "MT":[],
               "REDACAO":[]}

n_samples = 100000
sample = train_df.sample(n_samples)
    
for key, ys in tqdm(y_structure.items()):
    filtered_sample, missed_exam = presence_filter(sample, key)
    X = filtered_sample.drop(columns=target_columns+drop_columns)
    y = filtered_sample.filter(regex=f"NU_NOTA_{key}")
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state=42)
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)    
    search.fit(X_train, y_train)
    dump(search, f"models/model_{key}.joblib") 
    y_train_hat = search.predict(X_train)
    y_test_hat = search.predict(X_test)
    ys.extend([y_train, y_test, y_train_hat, y_test_hat])
    
mean_train = []
mean_test = []
for key, ys in tqdm(y_structure.items()):
    train_error = mean_squared_error(ys[0], ys[2], squared=False)
    mean_train.append(train_error)
    test_error = mean_squared_error(ys[1], ys[3], squared=False)
    mean_test.append(test_error)
    print(key)
    print(f"Train: {train_error}")
    print(f"Test: {test_error}\n")
    

print(np.mean(mean_test))
print(np.mean(mean_train))
