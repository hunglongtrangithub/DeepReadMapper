# DeepReadMapper (DRM)

DeepReadMapper is a deep learning-based gene alignment tool that uses vector similarity search to efficiently locate similar DNA sequences. It employs a pre-trained BiLSTM model to convert sequences into 128-dimensional embeddings, then performs fast approximate nearest neighbor search using HNSW indexing.

## Installation

Make sure [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install) is available.

1. Create conda environment

   ```bash
   conda env create -f environment.yml
   conda activate DeepReadMapper
   ```

2. Install external libraries

   ```bash
   bash setup_submodule.sh
   ```

3. Build the project

   ```bash
   zig build
   ```

All of the executables are in `zig-out/bin` directory.

## Usage

**Note**: The code only works on Linux. You will get compile errors on Windows or MacOS.

1. Index

```bash
hnswpq_index <ref_seq.txt> <index_prefix> <ref_len> [OPTIONS]
```

**Required Arguments:**

- `ref_seq.txt`: Path to reference file. Can be FASTA/txt/npy format.
- `index_prefix`: The prefix to the index folder. The index file and config.txt will be saved here.
- `ref_len`: Length of reference sequences.

**Optional Arguments:**

- `-s, --stride`: Stride for product quantization. Default: 1 (dense index)
- `-p, --M_pq`: Number of sub-vectors for product quantization. Default: 8
- `-b, --nbits`: Number of bits for each sub-vector (8, 10, or 12). Default: 8
- `-m, --M_hnsw`: Number of connections for each node in HNSW graph. Default: 16
- `-e, --EFC`: Size of dynamic list for HNSW graph construction. Default: 200
- `-h, --help`: Show help message

2. Search

```bash
pipeline <index_prefix> <query_file> <ref_file> [OPTIONS]
```

**Required Arguments:**

- `index_prefix`: Path to index folder containing .index file and config.txt
- `query_file`: Query sequences file (FASTQ/FASTA/TXT) or pre-computed embeddings (.npy)
- `ref_file`: Reference sequences file (FASTA/TXT)

**Optional Arguments:**

- `-e, --EF`: HNSW search parameter (higher = better accuracy, slower speed). Default: 128
- `-k, --K`: Number of nearest neighbors to return. Default: 128
- `-c, --K_clusters`: Number of clusters (only for sparse index with stride > 1). Default: varies
- `-o, --output_dir`: Output directory for results. Default: current directory
- `-d, --dynamic`: Load reference sequences dynamically (saves memory for large references)
- `-s, --streaming`: Use streaming output to SAM file (currently disabled)
- `-h, --help`: Show help message

## Sample usage

1. Create index on Ecoli 150 (`tests/ecoli_150.fna`):

```bash
./zig-out/bin/hnswpq_index ./tests/ecoli.fna ecoli_150_index 150
```

With custom parameters:

```bash
./zig-out/bin/hnswpq_index ./tests/ecoli.fna ecoli_150_index 150 --stride 2 --M_pq 16 --nbits 10
```

2. Perform search on Ecoli 150 queries (`tests/ecoli_150.fastq`):

```bash
./zig-out/bin/pipeline ecoli_150_index ./tests/ecoli_150.fastq ./tests/ecoli.fna
```

With custom parameters:

```bash
./zig-out/bin/pipeline ecoli_150_index ./tests/ecoli_150.fastq ./tests/ecoli.fna --EF 256 --K 64 --dynamic
```

The results will be saved in the current directory by default. There will be 2 numpy files: `indices.npy` and `distances.npy`.

**Note**: You can also modify `includes/utils/config.hpp` to change default parameters such as number of threads, batch sizes, and other settings.
