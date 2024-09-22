
import os
import pathlib
import opendatasets as od

def download_datasets_from_kaggle(dataset_url:str, dataset_dir:str)-> None:
    # Dowload the dataset
    data_dir = pathlib.Path(dataset_dir)
    data_dir.mkdir(exist_ok=True)
    if not os.path.isdir(dataset_dir):
        od.download(dataset_url)
    