## Notes
`modelling/outputs/100K_sample` returns the results reported in our report.

## Brief repository structure

### `CRISP_DM_Report.pdf` - final report.

### `modelling`:
- `embedding_generator.py` script to generate embeddings for the files;
- `tune_hdbscan.py` runs hyperparameter search on the embeddings;
- `clustering.py` runs clustering on the embeddings;
- `utils/` util functions for loading models and plotting visualizations;
- `labelling.py` runs labelling for the clusters generated during clustering phase;
- `config.yaml` sets the necessary config for the pipeline.

### `EDA`:
- `EDA.ipynb` contains the EDA with general and advanced data insights acquired.

### `file_unpacking`:
- `data_unpack.py` contains data unpacking techniques for large files;
- `data_unpack_multiprocess.py` contains data unpacking techniques for large files with multiprocessing.


## Reproducibility.
To reproduce the results in  `modelling/outputs/100K_sample`:
- Run `python file_unpacking/data_unpack_multiprocess.py --sample_size 1000`
- `cd modelling`
- On a separate terminal, instantiate an SGLANG server with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server   --model-path Qwen/Qwen3-8B   --port 30000  --host 0.0.0.0  --mem-fraction-static 0.7`
- Run `chmod +x pipeline.sh`
- Run `./pipeline.sh`
All the results are generated automatically. It's advised to run the pipeline on a small dataset size, like `1000` to not run into memory issues.
