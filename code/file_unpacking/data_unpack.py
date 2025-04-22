# Import necessary dependencies
import dask.bag as db
from dask.diagnostics import ProgressBar
import pandas as pd
import json

if __name__ == "__main__":
    # read the metadata of the main dataset
    docs = db.read_text("/home/viper/Data_science/DM_Project/ignore_folder/data/arxiv-metadata-oai-snapshot.json").map(json.loads)

    # Transform the main dataset into the dataframe
    trim = lambda x: {
        'id': x['id'],
        'title': x['title'],
        'category':x['categories'].split(' '),
        'abstract':x['abstract'],
        'authors':x['authors'],
        'authors_parsed':x['authors_parsed'],
        'update_date':x['update_date'],
    }

    with ProgressBar():
        docs_df = (
            docs.map(trim).compute(scheduler="processes")
        )

    docs_df.to_csv("ignore_folder/data/docs_df.csv", index=False)