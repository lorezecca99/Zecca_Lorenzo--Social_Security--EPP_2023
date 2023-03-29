import pandas as pd

def import_dataset(url):
    """Function to import the dataset as a csv file."""
    dataset=pd.read_csv(url)
    return dataset