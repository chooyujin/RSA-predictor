# RSA predictor

BERT-based architecture for relevant solvent accessibility (RSA) prediction

---

- [Installation](#installation)
- [Inference](#inference)

--- 

## Installation

```bash
conda create -n rsa_predictor python=3.10 -y
conda activate rsa_predictor

# CPU-only version
conda install pytorch torchvision torchaudio cpuonly -c pytorch

pip install transformers tqdm numpy sentencepiece protobuf huggingface_hub
```

For GPU environments, please install a PyTorch build that matches your CUDA version by following the official PyTorch installation guide


--- 

## Inference

```bash
python inference.py \
  --fasta input.fasta \
  --model model/RSA-predictor_params.pt \
  --gpu 0 \
  --batch_size 64 \
  --output prediction.csv
```

### Arguments

| Argument | Description |
|---|---|
| `--fasta` | Input FASTA file containing one or more protein sequences. |
| `--model` | Path to the trained model checkpoint. |
| `--gpu` | GPU device ID. Use `-1` to run on CPU. Default: `0`. |
| `--batch_size` | Number of sequences processed per batch. Default: `4`. |
| `--output` | Output CSV file name. Default: `prediction.csv`. |

### Input format

The input must be provided in FASTA format. For example:

```fasta
>Protein1
MKTAYIAKQRQISFVKSHFSRQDILDLWQ
>Protein2
GATPQDLNTMLNTVGSQARFVRASCP
```

### Output format

The prediction results are saved as a CSV file with the following columns:

| Column | Description |
|---|---|
| `Protein` | FASTA header |
| `Position_1based` | Residue position using 1-based indexing |
| `Pred_RSA` | Predicted RSA value for the residue |

Example output:

```csv
Protein,Position_1based,Pred_RSA
Protein1,1,0.1342
Protein1,2,0.1028
Protein1,3,0.0875
```

### Notes

- Use `--gpu -1` for CPU inference.
- GPU inference is recommended for faster embedding generation and prediction.
