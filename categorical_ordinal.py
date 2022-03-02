import pandas as pd

def get_categorical_ordinal_columns(df:pd.DataFrame):
    
    tp_more_missing = ["TP_DEPENDENCIA_ADM_ESC", 
                       "TP_LOCALIZACAO_ESC", 
                       "TP_SIT_FUNC_ESC"]

    tp_categorical_nominal = ["TP_SEXO", 
                              "TP_ESTADO_CIVIL", 
                              "TP_COR_RACA", 
                              "TP_NACIONALIDADE"]

    questions_categorical_ordinal = ["Q001", "Q002", "Q003", "Q004", 
                                     "Q005", "Q006", "Q007", "Q008", 
                                     "Q009", "Q010", "Q011", "Q012", 
                                     "Q013", "Q014", "Q015", "Q016", 
                                     "Q017", "Q019", "Q022", "Q024"]

    tp_excluded = tp_more_missing + tp_categorical_nominal
    tp_categorical_ordinal = (df.filter(regex="TP_")
                                .drop(columns=tp_excluded)
                                .columns
                                .tolist())
    categorical_ordinal = tp_categorical_ordinal + questions_categorical_ordinal
    
    return categorical_ordinal