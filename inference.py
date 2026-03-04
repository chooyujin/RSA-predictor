import argparse
import torch
from src.models.Model import TransformerRSA
from src.features.process import pad_batch, build_token_list
from src.features.esm_embedder import ESM2Embedder
from tqdm import tqdm 
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Run RSA-predictor inference from FASTA")

    parser.add_argument("--fasta",type=str,required=True,help="Input FASTA file path")
    parser.add_argument("--model",type=str,required=True,help="Model checkpoint path")
    parser.add_argument("--gpu",type=int,default=0,help="GPU id (-1 for CPU)")
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--output",type=str,default="prediction.csv",help="Output file")

    return parser.parse_args()

def parse_fasta(file_path):

    sequences = {}
    current_header = None
    current_seq = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue

            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq).upper()

                current_header = line[1:]
                current_seq = []

            else:
                current_seq.append(line)

        if current_header is not None:
            sequences[current_header] = "".join(current_seq).upper()

    if len(sequences) == 0:
        raise ValueError("No sequences found in FASTA")

    return sequences

def load_model(model_path:str, device: str="cpu"):
    ckpt = torch.load(model_path, map_location=device)

    model = TransformerRSA(vocab_size=21, d_model=256, 
                        nhead=4, num_layers=3, dim_feedforward=758,esm_dim=1280).to(device)
    
    model.load_state_dict(ckpt)
    model.eval()
    return model 

@torch.no_grad()
def run_inference(args):
    device = torch.device(f"cuda:{args.gpu}") if (args.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    print("Using Device: ", device)

    seqs = parse_fasta(args.fasta)
    
    model = load_model(args.model, device)
    embedder = ESM2Embedder(device=device)

    ids = list(seqs.keys())
    seq_list = [seqs[i] for i in ids]

    preds_by_id = {}

    batch_size = args.batch_size
    for i in tqdm(range(0, len(ids), batch_size)):
        batch_ids = ids[i:i+batch_size]
        batch_seqs = seq_list[i:i+batch_size]

        batch_tokens = [build_token_list(s) for s in batch_seqs]
        batch_esm = embedder.embed_batch(batch_seqs)
        token_np, esm_np,mask_np, lengths = pad_batch(batch_tokens, batch_esm)
        
        tokens = torch.from_numpy(token_np).to(device)
        esm_feat = torch.from_numpy(esm_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)

        assert tokens.shape[:2] == esm_feat.shape[:2] == mask.shape[:2]
        assert torch.all(tokens[~mask] == 0)

        out = model(tokens, esm_feat, mask).detach().cpu().numpy()

        for j, pid in enumerate(batch_ids):
            L = lengths[j]
            preds_by_id[pid] = out[j, :L].astype(float).tolist()
    
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Protein", "Position_1based", "Pred_RSA"])
        for pid in ids:
            pred = preds_by_id[pid]
            for pos, val in enumerate(pred, start=1):
                w.writerow([pid, pos, val])
    
    print("Saved:", args.output)


if __name__ == '__main__':
    
    args = parse_args()
    run_inference(args)