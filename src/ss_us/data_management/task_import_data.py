import pandas as pd
import pytask


from ss_us.config import SRC
from ss_us.data_management.import_data import import_dataset
    
@pytask.mark.produces(SRC / "data"/"dataset.csv")
def task_import_dataset(produces):
    """Clean the data (Python version).
    This task allows to import the merged datasets from 2014 to 2018 
    concering wages and hours worked in the US. To save time, we may import the 
    dataset only for 2018 just to prove that the code is running and the dataset has been 
    imported, saved and then correctly cleaned."""
    #data_info = read_yaml(depends_on["data_info"])
    #url = 'https://www.dropbox.com/s/a0re73x92nl8ama/dataset_14_18.csv?dl=1'
    url18='https://www.dropbox.com/s/3u4nzej8r1tsi78/dataset_18.csv?dl=1'
    #dataset=import_dataset(url)
    dataset18=import_dataset(url18)
    dataset18.to_csv(produces)
    return produces