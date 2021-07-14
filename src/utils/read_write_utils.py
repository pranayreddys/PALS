import pandas
import os
def read_data(filepath: str, format: str):
    is_file(filepath)
    if format == "csv":
        return pandas.read_csv(filepath)

def is_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)