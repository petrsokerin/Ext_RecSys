import numpy as np
import torch
import torch.nn as nn

from src.aggregation import BaseAggregation

class SASRec(BaseAggregation):
    def __init__(self,
        num_items,
        embedding_size=64,
        num_heads=2,
        num_blocks=2,
        dropout_rate=0.2,
        max_len=200,
        final='head',
        ext_flag=False
    ):
        # super(SASRec, self).__init__()
        super().__init__()
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.ext_flag = ext_flag
        self.final = final

        if final not in ['head', 'items']:
            raise ValueError("final should be head or items")

        self.item_emb = nn.Embedding(num_items + 1, embedding_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=num_heads,
                dim_feedforward=embedding_size,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, num_items)


    def get_internal_embeddings(self, input_seqs, pad_mask):
        batch_size, seq_len = input_seqs.size()

        # Position encoding
        positions = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Item and position embedding
        item_emb = self.item_emb(input_seqs)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)
        #print(x.shape, pad_mask.unsqueeze(-1).shape)
        x *= pad_mask.unsqueeze(-1)

        # Transformer encoder
        mask = self.generate_square_subsequent_mask(seq_len).to(input_seqs.device)
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.layer_norm(x)
        return x
    
    def predict_scores(self, user_embeddings):
        if self.final == 'head':
            return self.head(user_embeddings)
        elif self.final == 'items':
            # print(user_embeddings.shape, self.item_emb.weight[1:].transpose(-2, -1).shape)
            output = user_embeddings @ self.item_emb.weight[1:].transpose(-2, -1)
            return output


    def forward(self, input_seqs, pad_mask, timestamps=None):
        internal_emb = self.get_internal_embeddings(input_seqs, pad_mask)

        if self.ext_flag:
            output = self.external_forward(internal_emb, timestamps)
        else:
            output = self.predict_scores(internal_emb)

        # print(output.shape)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    