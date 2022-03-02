import pandas as pd

def get_categorical_nominal_columns(df:pd.DataFrame):
    sg_many_missing = "SG_UF_ESC"
    sg_columns = (df.filter(regex="SG_")
                    .drop(columns=sg_many_missing)
                    .columns
                    .tolist())
    
    in_columns = df.filter(regex="IN_").columns.tolist()
    tp_columns = ["TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA", "TP_NACIONALIDADE"]
    questions_columns = ["Q018", "Q020", "Q021", "Q023", "Q025"] 
    
    categorical_nominal = in_columns + tp_columns + questions_columns + sg_columns
    
    return categorical_nominal