import pandas

def read_data(filepath: str, format: str):
    if format == "csv":
        return pandas.read_csv(filepath)