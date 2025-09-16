from abc import ABC, abstractmethod
import torch
import numpy as np
from torch import nn
from src.model.simple_nn import SimpleNN

class BaseAggregation(ABC, nn.Module):
    def __init__(
        self, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs) 
        self.ext_flag = False

    # @property
    # @abstractmethod
    # def head(self) -> nn.Module:
    #     pass

    # @property
    # @abstractmethod
    # def embedding_size(self) -> int:
    #     pass

    # @property
    # @abstractmethod
    # def num_items(self) -> int:
    #     pass

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

        if head_method == "replace":
            self.ext_head = nn.Linear(self.embedding_size * 2, self.num_items + 1)
        elif head_method == "over":
            self.ext_head = nn.Linear(self.embedding_size, self.num_items + 1)
            self.over_head = nn.Linear(2 * (self.num_items + 1), self.num_items + 1)
        else:
            raise ValueError("head_method should be replace or over")


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
    
    def aggregate_external_embedding(self, internal_emb, external_context, ext_times):
        bs, seq_len, internal_emb_size = internal_emb.shape 
        all_external_context = external_context.reshape(bs*seq_len, self.n_ext_users, -1)
        all_internal_emb = internal_emb.reshape(bs*seq_len, -1)
        all_ext_times = ext_times.reshape(bs*seq_len, self.n_ext_users, -1)

        all_agg_embeddings = []
        for i in range(bs*seq_len):
            external_context, internal_emb, ext_times = all_external_context[i], all_internal_emb[i], all_ext_times[i]

            if self.agg_type == "mean":
                agg_embeddings = torch.Tensor(torch.mean(external_context, axis=0))

            elif self.agg_type == "max":
                agg_embeddings = torch.Tensor(torch.max(external_context, axis=0))

            elif "attention" in self.agg_type:
                external_context = torch.Tensor(external_context).to(internal_emb.device)

                if "learnable_attention" in self.agg_type:
                    if not self.learnable_attention_matrix:
                        raise ValueError("Learnable attention matrix wasn't initialized!")
                    external_context_prep = self.learnable_attention_matrix(external_context)

                elif self.agg_type == "symmetrical_attention":
                    if not self.learnable_attention_matrix:
                        raise ValueError("Learnable attention matrix wasn't initialized!")
                    external_context_prep = self.learnable_attention_matrix(external_context)
                    user_emb = self.learnable_attention_matrix(user_emb)

                elif self.agg_type == "kernel_attention":
                    if not self.attention_kernel:
                        raise ValueError("Attention kernel wasn't initialized!")
                    external_context_prep = self.attention_kernel(external_context)
                    user_emb = self.attention_kernel(user_emb)

                else: external_context_prep = external_context

                dot_prod = external_context_prep @ user_emb.unsqueeze(0).transpose(1, 0)
                softmax_dot_prod = nn.functional.softmax(dot_prod, 0)

                if "attention_hawkes" in self.agg_type:
                    times = torch.Tensor((times - ext_times) * self.exp_param).to(user_emb.device)
                    times = times.unsqueeze(-1)
                    time_part = torch.exp(times)
                    softmax_dot_prod = softmax_dot_prod * time_part

                agg_embeddings = (softmax_dot_prod * external_context).sum(dim=0)

            elif self.agg_type == "exp_hawkes":
                agg_embeddings = np.mean(external_context * np.exp((times - ext_times) * self.exp_param).reshape(-1,1), axis=0)
                agg_embeddings = torch.Tensor(agg_embeddings).to(user_emb.device)
                
            elif "hawkes" in self.agg_type:
                if not self.hawkes_nn:
                    raise ValueError("Hawkes NN wasn't initialized!")
                
                external_context = torch.Tensor(external_context).to(user_emb.device)
                concated = torch.cat((external_context, user_emb.tile((len(external_context), 1))), axis=1)
                emb_part = self.hawkes_nn(concated)
                times = torch.Tensor((times - ext_times) * self.exp_param).to(user_emb.device)
                times = times.unsqueeze(-1)
                if self.agg_type == "learnable_hawkes":
                    if not self.hawkes_time_nn:
                        raise ValueError("Hawkes NN for time wasn't initialized!")
                    time_part = self.hawkes_time_nn(times)
                elif self.agg_type == "exp_learnable_hawkes":
                    time_part = torch.exp(times)
                else:
                    raise ValueError("Unsupported pooling type.")
                agg_embeddings = torch.sum(emb_part * time_part, axis=0) 

            else:
                raise ValueError("Unsupported pooling type.")
            all_agg_embeddings.append(agg_embeddings)

        all_agg_embeddings = torch.stack(all_agg_embeddings).reshape(bs, seq_len, internal_emb_size)
        device = next(self.parameters()).device
        return all_agg_embeddings.to(device)

    

    def external_forward(self, internal_emb, timestamps):
        external_context, ext_times = self.get_external_embeddings(timestamps)
        external_context, ext_times = external_context.to(internal_emb.device), ext_times.to(internal_emb.device)
        external_emb = self.aggregate_external_embedding(internal_emb, external_context, ext_times)

        if self.head_method == "replace":
            extended_data = torch.cat([internal_emb, external_emb], dim=2)
            output = self.ext_head(extended_data)
        elif self.head_method == "over":
            internal_output = self.head(internal_emb)
            ext_output = self.ext_head(external_emb)
            extended_data = torch.cat([internal_output, ext_output], dim=2)
            output = self.over_head(extended_data)

        return output
