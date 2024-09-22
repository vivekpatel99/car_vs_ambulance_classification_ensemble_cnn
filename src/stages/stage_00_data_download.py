"""_summary_
"""
import os
import argparse
import pathlib
import opendatasets as od

from utils import utils

def data_download(config_path:str) -> None:
    """_summary_

    Args:
        config_path (str): _description_
    """
    utils.download_datasets_from_kaggle(config_path=config_path)



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser() 
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_download(config_path=args.config)