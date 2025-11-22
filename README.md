# DeepReadMapper (DRM)

DeepReadMapper is a deep learning-based gene alignment tool that uses vector similarity search to efficiently locate similar DNA sequences. It employs a pre-trained BiLSTM model to convert sequences into 128-dimensional embeddings, then performs fast approximate nearest neighbor search using HNSW indexing.

## Installation

1. Create conda environment

    ```bash
    conda create -f environment.yml
    conda activate DeepReadMapper
    ```

2. Install external libraries

    ```bash
    bash setup_submodule.sh
    ```

3. Build the project

    ```bash
    mkdir build && cd build
    cmake ..
    make -j32
    ```

## Usage

1. Index 
```bash
./hnswpq_index <ref_seq.txt> <index_prefix> <ref_len> [stride] [M_pq] [nbits] [M_hnsw] [EFC]
```

- `ref_seq.txt`: Path to reference file. Can be FASTA/txt/npy format.
- `index_prefix`: The prefix to the index folder. The index file and config.txt will be saved here.
- `ref_len`: Length of reference sequences.
- `stride`: (Optional) Stride for product quantization. Default: 1 (dense index)
- `M_pq`: (Optional) Number of sub-vectors for product quantization. Default: 8
- `nbits`: (Optional) Number of bits for each sub-vector. Default: 8
- `M_hnsw`: (Optional) Number of connections for each node in HNSW graph. Default: 16
- `EFC`: (Optional) Size of dynamic list for HNSW graph construction. Default: 200

2. Search

```bash
./pipeline <index_prefix> <query_seqs.fastq> <ref_seqs.fasta> [EF] [K] [K_clusters] [output_dir] [use_dynamic] [use_streaming]
```

- `index_prefix`: The prefix to the index folder. Contains the index file and config.txt
- `query_seqs.fastq`: Path to query file. Can be FASTQ/txt/npy format.
- `ref_seqs.fasta`: Path to reference file. Can be FASTA/txt format.
- `EF`: (Optional) HNSW exploration index. Higher means better accuracy but slower speed. Default: 128
- `K`: (Optional) Number of returned similar sequences. K <= EF. Default: 128
- `K_clusters`: (Optional) Number of clusters to use, ONLY applicable in sparse index (stride > 1). For dense index, K_clusters = K.
- `output_dir`: (Optional) Folder to save output files. Default: current folder.
- `use_dynamic`: (Optional) Whether to use dynamic sequence lookup in postprocessing step, which takes more time but saves memory. Postprocessing isn't applicable for dense index. Default: 0 (False)
- `use_streaming`: (Optional) Write output directly to disk after processing each query. For now, it is disable as SAM output is not supported. Default: 0 (False)

## Sample usage

1. Create index on Ecoli 150 (`tests/ecoli_150.fna`):

```bash
./build/hnswpq_index tests/ecoli_150.fna ecoli_150_index 150
```

2. Perform search on Ecoli 150 queries (`tests/ecoli_150.fastq`):

```bash
./build/pipeline ecoli_150_index tests/ecoli_150.fastq tests/ecoli_150.fna
```

The results will be saved in the current directory by default. There will be 2 numpy files: `indices.npy` and `distances.npy`.

*Note* You can also modify `includes/utils/config.hpp` to change default parameters such as number of threads, batch sizes, and other settings.
