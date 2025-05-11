import json
import pandas as pd
from multiprocessing import Pool, cpu_count
import os
import argparse
import random

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
    parser = argparse.ArgumentParser(description='Process arXiv metadata and optionally create a sample.')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Size of random sample to take (optional)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')
    args = parser.parse_args()
    file_path = "../ignore_folder/data/arxiv-metadata-oai-snapshot.json"

    # Adjust based on RAM amount
    chunk_size = 1000
    pool = Pool(cpu_count())

    all_data = []
    for chunk in read_in_chunks(file_path, chunk_size):
        processed_chunk = pool.map(process_chunk, [chunk])
        all_data.extend(processed_chunk[0])

    pool.close()
    pool.join()

    
    docs_df = pd.DataFrame(all_data)
    
    if args.sample_size is not None:
        random.seed(args.seed)
        docs_df = docs_df.sample(n=args.sample_size, random_state=args.seed)
        output_filename = f"../ignore_folder/data/sample_data_{args.sample_size}.csv"
    else:
        output_filename = "../ignore_folder/data/docs_df.csv"
    
    docs_df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

if __name__ == "__main__":
    main()