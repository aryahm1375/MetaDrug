import math
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import numpy as np
from pyhealth.datasets import SampleEHRDataset
from Models import CustomBaseModel as BaseModel
import copy
from collections import OrderedDict
import pickle

# VALID_OPERATION_LEVEL = ["visit", "event"]
import torch.optim as optim
import torch.nn.functional as F
class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

        self.attn_gradients = None
        self.attn_map = None

    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map

    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad

        # register_hook option allows us to save the gradients in backwarding

    def forward(self, query, key, value, mask=None, register_hook=False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        self.attn_map = attn  # save the attention map
        if register_hook:
            attn.register_hook(self.save_attn_grad)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, register_hook=False):

        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask, register_hook=register_hook))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)
class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLinear, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        #actory_kwargs = {'device': self.device, 'dtype': self.dtype}
        self.weight = nn.Parameter(torch.empty((output_dim, input_dim)))
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x, gu):
        #gu = self.preference_adapter(input)
        if gu == None:
            weight = self.weight
            return F.linear(x, self.weight, self.bias)#self.bias + x@self.weight#self.weight(x)#self.bias + x@self.weight#F.linear(x, weight, self.bias)
        else:
            weight = self.weight * gu.unsqueeze(1)

            return F.linear(x, weight, self.bias)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.user = CustomLinear(hidden_dim, hidden_dim)
        self.fc1 = CustomLinear(2*input_dim, hidden_dim)
        self.fc2 = CustomLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.preference_adapter = PreferenceSpecificAdapter(hidden_dim, hidden_dim)


    def forward(self, last_visit, user):
        user_x = self.user(user, None)
        gu = self.preference_adapter(user)
        user_visit = torch.cat((user_x, last_visit), dim = -1)
        x = self.fc1(user_visit, gu)
        x = self.relu(x)
        x = self.fc2(x, None)
        return x

class TransformerLayer(nn.Module):

    def __init__(self, feature_size, heads=1, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self, x: torch.tensor, mask: Optional[torch.tensor] = None, register_hook=False
    ) -> Tuple[torch.tensor, torch.tensor]:

        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask, register_hook)
        emb = x
        cls_emb = x[:, :, :].sum(1)
        return emb, cls_emb

class PreferenceSpecificAdapter(nn.Module):
    def __init__(self, preference_dim, hidden_dim):
        super(PreferenceSpecificAdapter, self).__init__()
        self.Wg = nn.Parameter(torch.randn(preference_dim, hidden_dim))
        self.bg = nn.Parameter(torch.randn(hidden_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, uo):
        # Compute preference-specific gate
        gu = self.sigmoid(torch.matmul(uo, self.Wg) + self.bg)
        return gu
class Transformer(BaseModel.BaseModel):

    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            pretrained_emb: str = None,
            embedding_dim: int = 128,
            timestep = 5,
            visit_adaptation = True,
            user_adaptation = True,
            **kwargs
    ):
        super(Transformer, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim

        # validate kwargs for Transformer layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")
        self.visit_adaptation = visit_adaptation
        self.user_adaptation = user_adaptation
        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        self.status = True
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()
        self.timestep = timestep
        self.lr = 1e-2
        self.fast_weights = copy.deepcopy(OrderedDict())
        #self.retain_fc = nn.Linear(len(self.feature_keys) * self.embedding_dim,  self.embedding_dim)
        #self.retain = nn.ModuleDict()
        #for feature_key in feature_keys:
        #    self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, **kwargs)
        # add feature transformation layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Transformer only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)
        self.transformer = nn.ModuleDict()
        self.local_update_target_weight_name = ['user.weight', 'user.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'preference_adapter.Wg', 'preference_adapter.bg']
        self.visit_level_weight_name = ['user.weight', 'user.bias','fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'preference_adapter.Wg', 'preference_adapter.bg']
        self.user_level_weight_name =['user.weight', 'user.bias']

        output_size = self.get_output_size(self.label_tokenizer)
        self.output_size = output_size
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim * self.timestep, self.embedding_dim)
        self.output = nn.Linear(self.embedding_dim, output_size)
        self.meta_optimizer = optim.SGD(self.parameters(), lr=1e-3)
        self.visit_transformer = nn.ModuleDict()
        for feature_key in feature_keys:
            self.visit_transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, **kwargs
            )

        self.user_transformer = nn.ModuleDict()
        for feature_key in feature_keys:
            self.user_transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, **kwargs
            )
        with open('/home/a053h213/PycharmProjects/pythonProject/LLM/sim_pat_dict.pkl', 'rb') as file:
            self.similar_user = pickle.load(file)
        '''self.MLP = nn.Sequential(
            nn.Linear(2 *self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.embedding_dim, self.output_size)
        )
        '''
        self.MLP = MLP(self.embedding_dim, self.embedding_dim, self.output_size)
        self.visit_fc = nn.Linear((len(self.feature_keys) - 1) * self.embedding_dim, self.embedding_dim)
        self.retain_fc = nn.Linear(2 * self.embedding_dim,  self.embedding_dim)
        self.store_parameters()

    def get_max_time(self, **kwargs):
        feature_key = 'drugs_hist'
        x = self.feat_tokenizers[feature_key].batch_encode_3d(
            kwargs[feature_key]
        )
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        x = self.embeddings[feature_key](x)
        return x.size()[1]


    def user_embedding(self, t=-1, similar=False, b=0, **kwargs):

        patient_emb = []
        for feature_key in ["conditions", "procedures"]:
            if not similar:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
            else:
                codes = [value[feature_key] for value in kwargs['similar_patients'][b]]
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    codes
                )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = x.view(x.size()[0], -1, x.size()[3])
            mask = torch.any(x, dim=2) != 0
            self.num_visits = torch.sum(mask, dim=1)
            _, x = self.user_transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        return self.retain_fc(patient_emb)

    def visit_embedding(self, index, b=0, similar=False, **kwargs):
        last_visit_embed = []
        for feature_key in ["conditions", "procedures"]:  # self.feature_keys:
            if not similar:
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
            else:
                codes = [value[feature_key] for value in kwargs['similar_patients'][b]]
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    codes
                )
            x = torch.tensor(x, dtype=torch.long, device=self.device)


            x = self.embeddings[feature_key](x)


            self.timestep = x.size()[1]

            input = x[:, index, :, :]

            mask = torch.any(input != 0, dim=2)
            output, x = self.visit_transformer[feature_key](input, mask, kwargs.get('register_hook'))
            last_visit_embed.append(output.sum(1))
        last_visit_embed = torch.cat(last_visit_embed, dim=1)
        visit_embed = self.visit_fc(last_visit_embed)

        return visit_embed
    def preference_adapter(self, patient_embedding, last_item_embed):
        concat_features = torch.cat((patient_embedding.clone(), last_item_embed.clone()), dim=-1)
        logit = self.MLP(concat_features)
        return logit

    def get_gu(self, input, batch_size):
        adapted = self.adapter_fc(input)
        return adapted.view(self.output_size, 2 * self.embedding_dim)
    def get_batch_gu(self, input, batch_size):
        adapted = self.adapter_fc(input)
        adapted_batch = adapted.view(batch_size, self.output_size, 2 * self.embedding_dim)
        return adapted_batch
    def get_user_adaptors(self, batch_size, input):
        adapted = self.adapter_fc(input)
        adapted_batch = adapted.view(batch_size, self.output_size, 2 * self.embedding_dim)
        return self.output_linear.weight.unsqueeze(0).repeat(128,1,1) * adapted
    def freeze_meta(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.adapter_fc.requires_grad = True
        self.output_linear.requires_grad = True
    def unfreeze_meta(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
    def patient_level_update(self, loss, model):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return model
    def output_prediction(self, patient_visit):
        self.output_linear.requires_grad = True
        return self.output_linear(patient_visit)
    def meta_update(self, meta_gradients):
        """Performs meta-update using accumulated gradients."""

        for p, g in zip(self.parameters(), meta_gradients):
            if g is not None:
                p.grad = g
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

    def get_meta_label(self, t, last = False,**kwargs):
        all_drug_hist = kwargs['drugs_hist']
        target = []
        for item in all_drug_hist:
            if len(item) <= t:
                target.append([])
            else:
                if item[t] == ['na']:
                    target.append([])
                else:
                    target.append(item[t])
        if last:
            self.prepare_labels(kwargs['drugs'], self.label_tokenizer)
        return self.prepare_labels(target, self.label_tokenizer)
    def store_parameters(self):
        self.keep_weight = copy.deepcopy(self.state_dict())
        self.weight_name = self.local_update_target_weight_name
        self.weight_len = len(self.keep_weight)
        self.fast_weights = copy.deepcopy(OrderedDict())
        self.meta_grad_init = [0 for _ in range(len(self.state_dict()))]

    def number_visits(self,**kwargs):
        n_visit_total= 0
        all_masks = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                kwargs[feature_key]
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            n_visit_total = x.size(1)
            x = self.embeddings[feature_key](x)

            x = torch.sum(x, dim=2)
            all_masks.append(torch.any(x, dim=2) != 0)
        result = all_masks[0]
        for tensor in all_masks[1:]:
            result = torch.logical_or(result, tensor)
        return torch.sum(result, dim = 1), n_visit_total
    def number_medical_codes(self, batch_size, **kwargs):
        num_codes = np.zeros(batch_size)
        for feature_key in ['conditions', 'procedures']:
            input = kwargs[feature_key]

            for i, batch in enumerate(input):
                b_n = 0
                for v in batch:
                    b_n += len(v)
                num_codes[i] += b_n
        return num_codes

    def jaccard_similarity(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        jaccard_sim = len(intersection) / len(union)

        return jaccard_sim

    def get_similar_users_embedding(self, b, **kwargs):
        user_id = kwargs['patient_id'][b]
        try:
            #print(self.similar_user.keys())
            number = len(self.similar_user[user_id])
            #print('Number', number)
            if number == 0:
                return []
            return self.user_embedding(similar=True, b=b, **kwargs)[1:1+number, :]
        except Exception as e:
            #print('Error', e)
            #return self.user_embedding(similar=True, b=b, **kwargs)[1:3, :]
            return []


    def get_similar_last_visit_embedding(self, b, **kwargs):
        try:
            user_id = kwargs['patient_id'][b]
            number = len(self.similar_user[user_id])
            if number == 0:
                return []
            return self.visit_embedding(-1, similar=True, b=b, **kwargs)[1:1+number, :]
        except:
            #return self.visit_embedding(-1, similar=True, b=b, **kwargs)[1:3, :]
            return []

    def get_similar_user_label(self, b, **kwargs):
        user_id = kwargs['patient_id'][b]
        labels = [value['drugs'] for value in kwargs['similar_patients'][b]]
        try:
            number = len(self.similar_user[user_id])
            if number == 0:
                return []

            return self.prepare_labels(labels, self.label_tokenizer)[1:1+number, :]
        except:
            #return self.prepare_labels(labels, self.label_tokenizer)[1:3, :]
            return []


    def get_cosine_similarity_socres(self, num_visit, **kwargs):
        last_visit = []
        all_other_visits = []
        tp_x = {}
        for feature_key in ['conditions', 'procedures']:
            tp_x[feature_key] = self.feat_tokenizers[feature_key].batch_encode_3d(
                kwargs[feature_key]
            )

        for i in range(0, num_visit):
            history = []
            for feature_key in ['conditions', 'procedures']:
                x = torch.tensor(tp_x[feature_key], dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                self.timestep = x.size()[1]

                input = x[:, i, :, :]

                mask = torch.any(input != 0, dim=2)
                output, x = self.visit_transformer[feature_key](input, mask, kwargs.get('register_hook'))
                if i == (num_visit - 1):
                    last_visit.append(x)
                else:
                    history.append(x)

            if not i == (num_visit - 1):
                all_other_visits.append(torch.cat(history, dim=1))
        last_visit = torch.cat(last_visit, dim=1)
        return all_other_visits, last_visit

    def visit_level_adaptation(self, all_meta_labels, b, patient_fast, all_visit_embeddings, last_visit_time):
        self.fast_weights = OrderedDict()
        visit_level_loss = []
        visit_level_grads = []
        with torch.enable_grad():
            weight_for_local_update = [self.MLP.state_dict()[key] for key in self.weight_name]
            for idx in range(0, 1):
                t_loss = []
                for t in range(0, last_visit_time - 1):
                    if torch.sum(all_meta_labels[t][b]).item() > 0:
                        logit = self.MLP(all_visit_embeddings[t][b], patient_fast[b])
                        t_loss.append(self.get_loss_function()(logit, all_meta_labels[t][b]))
                if len(t_loss) > 0:
                    loss = torch.stack((t_loss), dim=0).mean(0)
                    visit_level_loss.append(loss.item())
                    grad = torch.autograd.grad(loss, self.MLP.parameters(), create_graph=True, allow_unused=True)
                    '''for i in range(len(grad)):
                        if grad[i] is not None:
                            #torch.nn.utils.clip_grad_norm_(grad[i], 1.0)
                            torch.nn.utils.clip_grad_value_(grad[i], 1.0)
                    '''
                    grad_norms = [torch.norm(g) for g in grad if g is not None]
                    max_grad_norm = max(grad_norms)
                    if max_grad_norm > 1:
                        scaling_factor = 1 / max_grad_norm
                        grad = [g * scaling_factor if g is not None else None for g in grad]
                    visit_level_grads.append(np.sum([torch.sum(grad[i]).detach().cpu().numpy() for i in range(len(grad))]))
                    if idx == 0:
                        for i in range(len(grad)):
                            if self.weight_name[i] in self.visit_level_weight_name:
                                self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.lr * grad[i]

                            else:
                                self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
                    else:
                        for i in range(len(grad)):
                            if self.weight_name[i] in self.visit_level_weight_name:
                                self.fast_weights[self.weight_name[i]] = self.fast_weights[
                                                                             self.weight_name[i]] - self.lr * grad[i]
                            else:
                                self.fast_weights[self.weight_name[i]] = self.fast_weights[self.weight_name[i]]
                    for name, param in self.named_parameters():
                        if name.replace('MLP.', '') in self.fast_weights.keys():
                            with torch.no_grad():
                                param.data.copy_(self.fast_weights[name.replace('MLP.', '')].data.to(self.device))
        return visit_level_loss, visit_level_grads

    def clip_grads(self, grads, max_norm):
        clipped_grads = []
        for grad in grads:
            if grad is not None:
                clipped_grads.append(torch.clamp(grad, -max_norm, max_norm))
            else:
                clipped_grads.append(grad)
        return clipped_grads

    def user_level_adaptation(self, similar_user_embeddings, similar_visit_embedding, similar_user_label):
        user_level_loss = []
        user_level_grads = []
        self.fast_weights = OrderedDict()
        with torch.enable_grad():
            weight_for_local_update = [self.MLP.state_dict()[key] for key in self.weight_name]
            for idx in range(0, 1):
                loss_similar = []
                for similar_index in range(similar_user_embeddings.size()[0]):
                    logit = self.MLP(similar_visit_embedding[similar_index], similar_user_embeddings[similar_index])
                    loss_similar.append(self.get_loss_function()(logit, similar_user_label[similar_index]))
                loss = torch.stack((loss_similar), dim=0).mean(0)
                #print(loss.mean())
                user_level_loss.append(loss.item())
                grad = torch.autograd.grad(loss, self.MLP.parameters(), create_graph=True, allow_unused=True)
                grad_norms = [torch.norm(g) for g in grad if g is not None]
                max_grad_norm = max(grad_norms)
                if max_grad_norm > 1:
                    scaling_factor = 1 / max_grad_norm
                    grad = [g * scaling_factor if g is not None else None for g in grad]
                user_level_grads.append(np.sum([torch.sum(grad[i]).detach().cpu().numpy() for i in range(len(grad))]))
                if idx == 0:
                    for i in range(len(grad)):
                        if self.weight_name[i] in self.visit_level_weight_name:
                            self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - 1e-2 * grad[i]
                        else:
                            self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
                else:
                    for i in range(len(grad)):
                        if self.weight_name[i] in self.visit_level_weight_name:
                            self.fast_weights[self.weight_name[i]] = self.fast_weights[
                                                                         self.weight_name[i]] - 1e-2 * grad[i]
                        else:
                            self.fast_weights[self.weight_name[i]] = self.fast_weights[self.weight_name[i]]
                for name, param in self.named_parameters():
                    if name.replace('MLP.', '') in self.fast_weights.keys():
                        with torch.no_grad():
                            param.data.copy_(self.fast_weights[name.replace('MLP.', '')].data.to(self.device))
            return user_level_loss, user_level_grads

    def get_relevant_visits(self, b, num_visit, **kwargs):
        last_visit = []
        all_other_visits = []
        rel_times = []
        tp_x = {}
        for feature_key in ['conditions', 'procedures']:
            tp_x[feature_key] = self.feat_tokenizers[feature_key].batch_encode_3d(
                kwargs[feature_key]
            )
        for i in range(0, num_visit):
            history = []
            for feature_key in ['conditions', 'procedures']:
                if i == (num_visit-1):

                    current_visit = tp_x[feature_key][b][-1]
                    current_visit = [f"{feature_key}{str(item)}" for item in current_visit if not item == 0]
                    last_visit = last_visit + list(set(current_visit))
                else:
                    history_visit = tp_x[feature_key][b][i]
                    history_visit = [f"{feature_key}{str(item)}" for item in history_visit if not item == 0]
                    history = history + list(set(history_visit))
            if not  i == (num_visit-1):
                all_other_visits.append(history)
        for t in range(0, num_visit-1):
            if len(last_visit) > 0 and len(all_other_visits) > 0:
                if self.jaccard_similarity(last_visit, all_other_visits[t]) > 0.1:
                    rel_times.append(t)
        return rel_times

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        meta_grad = copy.deepcopy(self.meta_grad_init)
        n_visits, total = self.number_visits(**kwargs)
        last_visit = self.visit_embedding(-1, **kwargs)
        batch_size = last_visit.size()[0]
        number_of_medical_codes = self.number_medical_codes(batch_size, **kwargs)

        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        query_logits = []
        query_losses = []
        last_visit_time = self.timestep
        all_meta_labels = []
        patient_fast = self.user_embedding(**kwargs)
        all_visit_embeddings = []
        if not self.training and (self.status != self.training):
            self.iter = 0

        all_user_loss = []
        all_user_grads = []
        all_visit_loss = []
        all_visit_grads = []
        all_user_embeddings = []


        for t in range(0, last_visit_time):
            all_meta_labels.append(self.get_meta_label(t, **kwargs))
            all_visit_embeddings.append(self.visit_embedding(t, **kwargs))

        for b in range(0, batch_size):
            if not self.training:
                all_user_embeddings.append(patient_fast[b].cpu().numpy())
            if self.visit_adaptation:
                visit_level_loss, visit_level_grads = self.visit_level_adaptation(all_meta_labels, b, patient_fast, all_visit_embeddings, last_visit_time)
                all_visit_loss = all_visit_loss + list(visit_level_loss)
                all_visit_grads = all_visit_grads + list(visit_level_grads)
            if self.user_adaptation:
                similar_user_embeddings = self.get_similar_users_embedding(b, **kwargs)
                similar_visit_embedding = self.get_similar_last_visit_embedding(b, **kwargs)
                similar_user_label = self.get_similar_user_label(b, **kwargs)
                if not similar_user_embeddings == []:
                    user_level_loss, user_level_grads = self.user_level_adaptation(similar_user_embeddings, similar_visit_embedding,
                                          similar_user_label)
                    all_user_loss = all_user_loss + list(user_level_loss)
                    all_user_grads = all_user_grads + list(user_level_grads)
            query_patient = patient_fast[b]
            last_visit_query = all_visit_embeddings[-1][b] #inja bug dare baraye last visit
            query_logit = self.MLP(last_visit_query, query_patient)
            query_logits.append(query_logit)
            loss_q = self.get_loss_function()(query_logit, y_true[b])
            query_losses.append(loss_q)
            if self.user_adaptation or self.visit_adaptation:
                with torch.enable_grad():
                    for name, param in self.named_parameters():
                        if name.replace('MLP.', '') in self.local_update_target_weight_name:
                            with torch.no_grad():
                                param.data.copy_(self.keep_weight[name].data.to(self.device))
            if self.training:
                task_grad_test = torch.autograd.grad(loss_q, self.parameters(),allow_unused=True, retain_graph=True)
                for g in range(len(task_grad_test)):
                    if not task_grad_test[g] == None:
                        meta_grad[g] += task_grad_test[g].detach()
                    else:
                        meta_grad[g] = None

        y_prob = self.prepare_y_prob(torch.stack(query_logits))

        if not self.training:
            all_user_embeddings = np.array(all_user_embeddings)
            if self.visit_adaptation and not self.user_adaptation:
                np.save(f'ARRAYS/visits_{self.iter}.npy', all_user_embeddings)
                np.save(f'ARRAYS/n_codes_visits_{self.iter}.npy', np.array(number_of_medical_codes))
            elif not self.visit_adaptation and self.user_adaptation:
                np.save(f'ARRAYS/users_{self.iter}.npy', all_user_embeddings)
                np.save(f'ARRAYS/n_codes_users_{self.iter}.npy', np.array(number_of_medical_codes))
            elif self.visit_adaptation and self.user_adaptation:
                np.save(f'ARRAYS/full_{self.iter}.npy', all_user_embeddings)
                np.save(f'ARRAYS/n_codes_full_{self.iter}.npy', np.array(number_of_medical_codes))
            else:
                np.save(f'ARRAYS/no_adapt_{self.iter}.npy', all_user_embeddings)
                np.save(f'ARRAYS/n_codes_no_adapt{self.iter}.npy', np.array(number_of_medical_codes))

        self.status = self.training
        if not self.training:
            self.iter += 1
        print('loss/grad user', np.mean(all_user_loss), np.mean(all_user_grads))
        print('loss/grad visit', np.mean(all_visit_loss), np.mean(all_visit_grads))
        print('Query Loss:', torch.stack(query_losses).mean(0))
        results = {
            "loss": torch.stack(query_losses).mean(0),
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": 0,
            'num_visits': n_visits,
            'grads': meta_grad,
            'n_codes': torch.tensor(number_of_medical_codes).to(self.device)
        }

        return results

    def meta_update(self, meta_gradients):
        """Performs meta-update using accumulated gradients."""
        for p, g in zip(self.parameters(), meta_gradients):
            if g is not None:
                p.grad = g
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
