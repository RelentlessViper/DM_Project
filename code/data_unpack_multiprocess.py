import json
import pandas as pd
from multiprocessing import Pool, cpu_count
import os

def process_chunk(chunk):
    """Process a chunk of JSON lines."""
    data = []
    for line in chunk:
        record = json.loads(line)
        trimmed_record = {
            'id': record['id'],
            'title': record['title'],
            'category': record['categories'].split(' '),
            'abstract': record['abstract'],
            'authors': record['authors'],
            'authors_parsed': record['authors_parsed'],
            'update_date': record['update_date'],
        }
        data.append(trimmed_record)
    return data

def read_in_chunks(file_path, chunk_size=1000):
    """Lazy function to read a file in chunks."""
    with open(file_path, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def main():
    file_path = "/home/viper/Data_science/DM_Project/ignore_folder/data/arxiv-metadata-oai-snapshot.json"

    # Adjust based on RAM amount
    chunk_size = 1000
    pool = Pool(cpu_count())

    all_data = []
    for chunk in read_in_chunks(file_path, chunk_size):
        processed_chunk = pool.map(process_chunk, [chunk])
        all_data.extend(processed_chunk[0])

    pool.close()
    pool.join()

    # Convert the list of dictionaries to a dataframe
    docs_df = pd.DataFrame(all_data)
    docs_df.to_csv("/home/viper/Data_science/DM_Project/ignore_folder/data/docs_df.csv", index=False)

if __name__ == "__main__":
    main()