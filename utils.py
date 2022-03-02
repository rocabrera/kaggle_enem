import pandas as pd

def _fillna_target(df, presence, score):
    cond = df[presence] != 1
    df.loc[cond, score] = df.loc[cond, score].fillna(0)

def clean_target(df:pd.DataFrame):
    _fillna_target(df, "TP_PRESENCA_CN", "NU_NOTA_CN")
    _fillna_target(df, "TP_PRESENCA_CH", "NU_NOTA_CH")
    _fillna_target(df, "TP_PRESENCA_LC", "NU_NOTA_LC")
    _fillna_target(df, "TP_PRESENCA_MT", "NU_NOTA_MT")
    _fillna_target(df, "TP_STATUS_REDACAO", "NU_NOTA_REDACAO")


