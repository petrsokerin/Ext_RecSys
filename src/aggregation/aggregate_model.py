from abc import ABC, abstractmethod
import torch
import numpy as np
from torch import nn

class SimpleNN(nn.Module):
    """Fully-connected neural network model with ReLU activations"""
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int = None
    ) -> None:
        """Initialize method for SimpleNN.
        
        Args:
        ----
            input_size (int): Input size
            hidden_sizes (list[int]): Hidden sizes (if output_size is None then the last size is output size)
            output_size (int): Output
         """
        super().__init__()
        layers = nn.ModuleList([])
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        if output_size is not None:
            layers.append(nn.Linear(prev_size, output_size))
        self.layers = layers
    
    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = nn.functional.relu(out)
            out = layer(out)
        return out

class BaseAggregation(ABC, nn.Module):
    def __init__(
        self, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs) 
        self.ext_flag = False


    @abstractmethod
    def forward(self):
        pass


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print("Все слои сети заморожены.")


    def add_external_features(
        self, 
        time_list,
        time_to_embeddings,
        ext_features,
        head_method="replace",
        alpha=0.5,
        agg_type="mean",
        additional_config=None
    ):

        self.ext_flag = True
        self.freeze()

        self.head_method = head_method
        
        self.agg_type = agg_type
        self.additional_config = additional_config

        self.time_list = time_list
        self.time_to_embeddings = time_to_embeddings
        self.ext_embeddings = ext_features

        self.n_ext_users = self.ext_embeddings.shape[1]

        if head_method == "replace" and self.final=='head':
            self.ext_head = nn.Linear(self.embedding_size * 2, self.num_items)
        elif head_method == "over" and self.final=='head':
            self.ext_head = nn.Linear(self.embedding_size, self.num_items)
            self.over_head = nn.Linear(2 * (self.num_items), self.num_items)
        elif head_method == "adding" and self.final=='head':
            self.ext_head = nn.Linear(self.embedding_size, self.num_items)
            self.alpha=alpha
        elif head_method == "replace" and self.final=='items':
            raise ValueError("head_method replace can be combined with final items")
        elif head_method == "over" and self.final=='items':
            self.over_head = nn.Linear(2 * (self.num_items), self.num_items)
        elif head_method == "adding" and self.final=='items':
            self.alpha=alpha

        else:
            raise ValueError("head_method should be replace, over or adding")


        if agg_type == "learnable_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.embedding_size, self.embedding_size
            )
        elif agg_type == "symmetrical_attention":
            self.learnable_attention_matrix = nn.Linear(
                self.embedding_size, additional_config["hidden_size"]
            )
        elif agg_type == "kernel_attention":
            self.attention_kernel = SimpleNN(
                input_size=self.embedding_size,
                hidden_sizes=additional_config["hidden_sizes"]
            )
        elif agg_type == "exp_hawkes" :
            self.exp_param = additional_config["exp_param"] if (additional_config and "exp_param" in additional_config) else 1.
        elif agg_type == "attention_hawkes":
            self.exp_param = additional_config["exp_param"] if (additional_config and "exp_param" in additional_config) else 1.
        elif agg_type == "learnable_hawkes":
            self.hawkes_nn = SimpleNN(
                input_size=self.embedding_size * 2,
                hidden_sizes=additional_config["hawkes_nn"]["hidden_sizes"],
                output_size=self.embedding_size
            )
            self.hawkes_time_nn = SimpleNN(
                input_size=1,
                hidden_sizes=additional_config["hawkes_time_nn"]["hidden_sizes"],
                output_size=1
            )
        elif agg_type == "exp_learnable_hawkes":
            self.exp_param = additional_config["exp_param"] if (additional_config and "exp_param" in additional_config) else 1.
            self.hawkes_nn = SimpleNN(
                input_size=self.embedding_size * 2,
                hidden_sizes=additional_config["hidden_sizes"],
                output_size=self.embedding_size
            )
        elif agg_type == "learnable_attention_hawkes":
            self.exp_param = additional_config["exp_param"] if (additional_config and "exp_param" in additional_config) else 1.
            self.learnable_attention_matrix = nn.Linear(
                self.embedding_size, self.embedding_size
            )


    def get_external_embeddings(self, timestamps):
        bs, seq_len = timestamps.shape
        timestamps = timestamps.reshape(-1).cpu().detach().numpy()

        ext_ids = np.searchsorted(self.time_list, timestamps, side='right') - 1
        ext_context = torch.tensor(self.ext_embeddings[ext_ids], dtype=torch.float32).reshape(bs, seq_len, self.n_ext_users, self.embedding_size)
        ext_times = torch.tensor(self.time_to_embeddings[ext_ids], dtype=torch.float32).reshape(bs, seq_len, self.n_ext_users)
        return ext_context, ext_times
    
    def aggregate_external_embedding(self, internal_emb, external_context, ext_times, timestamps):
        bs, seq_len, n_ext_users, emb_size = external_context.shape 


        if self.agg_type == "mean":
            agg_embeddings = torch.mean(external_context, dim=2)

        elif self.agg_type == "max":
            agg_embeddings = torch.max(external_context, dim=2).values

        elif "attention" in self.agg_type:
            external_context = external_context.to(internal_emb.device)

            if "learnable_attention" in self.agg_type:
                if not self.learnable_attention_matrix:
                    raise ValueError("Learnable attention matrix wasn't initialized!")
                external_context_prep = self.learnable_attention_matrix(external_context)

            elif self.agg_type == "symmetrical_attention":
                if not self.learnable_attention_matrix:
                    raise ValueError("Learnable attention matrix wasn't initialized!")
                external_context_prep = self.learnable_attention_matrix(external_context)
                internal_emb = self.learnable_attention_matrix(internal_emb)

            elif self.agg_type == "kernel_attention":
                if not self.attention_kernel:
                    raise ValueError("Attention kernel wasn't initialized!")
                external_context_prep = self.attention_kernel(external_context)
                internal_emb = self.attention_kernel(internal_emb)

            else: external_context_prep = external_context

            dot_prod = external_context_prep @ internal_emb.unsqueeze(-2).transpose(-2, -1)

            if "attention_hawkes" in self.agg_type:
                times = torch.Tensor((timestamps - ext_times) * self.exp_param).to(internal_emb.device)
                times = times.unsqueeze(-1)
                time_part = torch.exp(times)
                dot_prod = dot_prod * time_part
            
            softmax_dot_prod = nn.functional.softmax(dot_prod, 2)
            agg_embeddings = (softmax_dot_prod * external_context).sum(dim=2)

        elif self.agg_type == "exp_hawkes":
            time_diff = timestamps.unsqueeze(-1) - ext_times
            exp_time_diff = torch.exp(time_diff * self.exp_param)
            emb_time_norm = external_context * exp_time_diff.unsqueeze(-1)
            agg_embeddings = torch.mean(emb_time_norm, dim=2)

            agg_embeddings = torch.Tensor(agg_embeddings).to(internal_emb.device)
            
        elif "hawkes" in self.agg_type:
            if not self.hawkes_nn:
                raise ValueError("Hawkes NN wasn't initialized!")
            
            external_context = torch.Tensor(external_context).to(internal_emb.device)

            concated = torch.cat((external_context, internal_emb.unsqueeze(2).tile((1, 1, external_context.shape[2], 1))), axis=-1)
            emb_part = self.hawkes_nn(concated)

            times = torch.Tensor((timestamps.unsqueeze(-1) - ext_times) * self.exp_param).to(internal_emb.device)
            times = times.unsqueeze(-1)
            if self.agg_type == "learnable_hawkes":
                if not self.hawkes_time_nn:
                    raise ValueError("Hawkes NN for time wasn't initialized!")
                time_part = self.hawkes_time_nn(times)
            elif self.agg_type == "exp_learnable_hawkes":
                time_part = torch.exp(times)
            else:
                raise ValueError("Unsupported pooling type.")
            agg_embeddings = torch.sum(emb_part * time_part, dim=2) 

        else:
            raise ValueError("Unsupported pooling type.")
        
        device = next(self.parameters()).device
        return agg_embeddings.to(device)
    

    def external_forward(self, internal_emb, timestamps):
        external_context, ext_times = self.get_external_embeddings(timestamps)
        external_context, ext_times = external_context.to(internal_emb.device), ext_times.to(internal_emb.device)
        timestamps = timestamps.to(internal_emb.device)
        external_emb = self.aggregate_external_embedding(internal_emb, external_context, ext_times, timestamps)

        # if self.head_method == "replace":
        #     extended_data = torch.cat([internal_emb, external_emb], dim=2)
        #     output = self.ext_head(extended_data)
        # elif self.head_method == "over":
        #     internal_output = self.predict_scores(internal_emb)
        #     ext_output = self.ext_head(external_emb)
        #     extended_data = torch.cat([internal_output, ext_output], dim=2)
        #     output = self.over_head(extended_data)

        if self.head_method == "replace" and self.final=='head':
            extended_data = torch.cat([internal_emb, external_emb], dim=2)
            output = self.ext_head(extended_data)

        elif self.head_method == "over" and self.final=='head':
            internal_output = self.predict_scores(internal_emb)
            ext_output = self.ext_head(external_emb)
            extended_data = torch.cat([internal_output, ext_output], dim=2)
            output = self.over_head(extended_data)

        elif self.head_method == "adding" and self.final=='head':
            internal_output = self.predict_scores(internal_emb)
            ext_output = self.ext_head(external_emb)
            output = self.alpha * ext_output + (1 - self.alpha) * internal_output

        elif self.head_method == "over" and self.final=='items':
            internal_output = self.predict_scores(internal_emb)
            ext_output = self.predict_scores(external_emb)
            extended_data = torch.cat([internal_output, ext_output], dim=2)
            output = self.over_head(extended_data)

        elif self.head_method == "adding" and self.final=='items':
            internal_output = self.predict_scores(internal_emb)
            ext_output = self.predict_scores(external_emb)
            output = self.alpha * ext_output + (1 - self.alpha) * internal_output
        return output
