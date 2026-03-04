import torch
from transformers import AutoTokenizer, AutoModel


class ESM2Embedder:

    def __init__(self, device):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t33_650M_UR50D"
        )

        self.model = AutoModel.from_pretrained(
            "facebook/esm2_t33_650M_UR50D"
        )

        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_batch(self, seq_list):

        emb_list = []

        for seq in seq_list:

            inputs = self.tokenizer(
                seq,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)

            outputs = self.model(**inputs, return_dict=True)
            hidden = outputs.last_hidden_state.squeeze(0)[1:-1]
            emb_list.append(hidden.cpu().half())

        return emb_list