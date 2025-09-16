import numpy as np
import torch
import torch.nn as nn

from src.model.aggregate_model import BaseAggregation

class SASRec(BaseAggregation):
    def __init__(self,
        num_items,
        embedding_size=64,
        num_heads=2,
        num_blocks=2,
        dropout_rate=0.2,
        max_len=200,
        ext_flag=False
    ):
        # super(SASRec, self).__init__()
        super().__init__()
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.ext_flag = ext_flag

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
        self.head = nn.Linear(embedding_size, num_items + 1)

    # @property
    # def head(self) -> nn.Module:
    #     return self.head

    # @property
    # def embedding_size(self) -> int:
    #     return self.embedding_size

    # @property
    # def num_items(self) -> int:
    #     return self.num_items

    def get_internal_embeddings(self, input_seqs):
        batch_size, seq_len = input_seqs.size()

        # Position encoding
        positions = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Item and position embedding
        item_emb = self.item_emb(input_seqs)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.dropout(x)

        # Transformer encoder
        mask = self.generate_square_subsequent_mask(seq_len).to(input_seqs.device)
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.layer_norm(x)
        return x


    def forward(self, input_seqs, timestamps=None):
        internal_emb = self.get_internal_embeddings(input_seqs)

        if self.ext_flag:
            output = self.external_forward(internal_emb, timestamps)
        else:
            output = self.head(internal_emb)

        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # def get_embeddings(self, input_seqs):
    #     batch_size, seq_len = input_seqs.size()

    #     positions = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
    #     positions = positions.unsqueeze(0).expand(batch_size, seq_len)

    #     item_emb = self.item_emb(input_seqs)
    #     pos_emb = self.pos_emb(positions)
    #     x = item_emb + pos_emb

    #     mask = self.generate_square_subsequent_mask(seq_len).to(input_seqs.device)
    #     for layer in self.encoder_layers:
    #         x = layer(x, mask)

    #     x = self.layer_norm(x)
    #     return x
    
    # def freeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = False
    #     print("Все слои сети заморожены.")
    
    # def add_external_features(self, ext_features, agg_type="mean", additional_config=None):
    #     self.ext_flag = True
    #     self.freeze()

    #     self.time_list = ext_features[0]
    #     self.ext_embeddings = ext_features[1]

    #     self.ext_head = nn.Linear(self.embedding_size * 2, self.num_items + 1)

    #     if agg_type == "learnable_attention":
    #         self.learnable_attention_matrix = nn.Linear(
    #             self.backbone_embd_size, self.backbone_embd_size
    #         )
    #     elif agg_type == "symmetrical_attention":
    #         self.learnable_attention_matrix = nn.Linear(
    #             self.backbone_embd_size, additional_config["hidden_size"]
    #         )
    #     elif agg_type == "kernel_attention":
    #         self.attention_kernel = SimpleNN(
    #             input_size=self.backbone_embd_size,
    #             hidden_sizes=additional_config["hidden_sizes"]
    #         )
    #     elif agg_type == "learnable_hawkes":
    #         self.hawkes_nn = SimpleNN(
    #             input_size=self.backbone_embd_size * 2,
    #             hidden_sizes=additional_config["hawkes_nn"]["hidden_sizes"],
    #             output_size=self.backbone_embd_size
    #         )
    #         self.hawkes_time_nn = SimpleNN(
    #             input_size=1,
    #             hidden_sizes=additional_config["hawkes_time_nn"]["hidden_sizes"],
    #             output_size=1
    #         )
    #     elif agg_type == "exp_learnable_hawkes":
    #         self.hawkes_nn = SimpleNN(
    #             input_size=self.backbone_embd_size * 2,
    #             hidden_sizes=additional_config["hidden_sizes"],
    #             output_size=self.backbone_embd_size
    #         )


    # def get_external_features(self, timestamps):
    #     bs, seq_len = timestamps.shape
    #     timestamps = timestamps.reshape(-1).cpu().detach().numpy()

    #     ext_ids = np.searchsorted(self.time_list, timestamps, side='right') - 1
    #     ext_context = self.ext_embeddings[ext_ids]
    #     ext_context = torch.tensor(ext_context, dtype=torch.float32).reshape(bs, seq_len, -1)
    #     return ext_context