import pandas as pd


def load_data(filepath):
    """Загружает датасет из CSV-файла"""
    return pd.read_csv(filepath)


def clean_data(df):
    """Очистка данных и приведение к стандартному виду(duration удаляем потому что происходит data leaking)"""
    df = df.copy()
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
    df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
    return df