import torch
import torch.nn as nn
import json
from torch.nn import functional as F
from torch.nn.modules import BCEWithLogitsLoss
from torch.utils.data import Dataset
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel

class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.classifier_intermediate_hidden_size = kwargs.get("classifier_intermediate_hidden_size", 1024)
        self.classifier_act_func = kwargs.get("classifier_act_func", "gelu")
        

class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQueryNER, self).__init__(config)
        
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout,
                                                       intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.hidden_size = config.hidden_size
        
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


#         # bert last_hidden_state输出
        sequence_heatmap = bert_outputs.last_hidden_state  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()
        # 实体起始、结束位置的推理
        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # 为序列中每个token所对应的实体，都需要计算它们起始和结束位置匹配的概率
        # [batch, seq_len, 1, hidden] => [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, 1, seq_len, hidden] => [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # 拼接两个矩阵最后一个维度的hidden [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits

class MultiNonLinearClassifier(nn.Module):
    
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        # 如果没有设置 intermediate_hidden_size, 就使用 hidden_size 作为中间层参数数量
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

class MRCNERDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length: int = 512):
        # 加载json语料文件数据
        self.all_data = json.load(open(json_path, encoding="utf-8-sig"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):

        data = self.all_data[item]
        tokenizer = self.tokenizer

        query = data["query"]  # 问题描述
        context = data["context"]  # 内容描述
        start_positions = data["start_position"]  # 实体起始位置索引集合
        end_positions = data["end_position"]  # 实体结束位置索引集合

        # end_positions = [ x + 1 for x in end_positions ]  
        # 原来数据的end_positions是标到最后一个字上，所以这里要往后推一个
        # 生成query和context的bert模型输入
        query_context_tokens = tokenizer(query, context, return_offsets_mapping=True)
        tokens = query_context_tokens.input_ids
        type_ids = query_context_tokens.token_type_ids
        offsets = query_context_tokens.offset_mapping
        
        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        offstart,offend = [],[]
        for token_idx in range(len(tokens)):
            # 跳过 query token 不处理
            if type_ids[token_idx] == 0:
                continue
            # token相对于原始文本的起始和结束位置索引
            token_start, token_end = offsets[token_idx]
            # 跳过 [CLS] 和 [SEP] 标记
            if token_start == token_end == 0:
                continue
            offstart.append(token_start)
            offend.append(token_end)
            # 实体在原始文本中的起始位置对应的token_index
            origin_offset2token_idx_start[token_start] = token_idx
            # 实体在原始文本中的结束位置对应的token_index
            origin_offset2token_idx_end[token_end] = token_idx
        
        # 语料中实体的起始、结束位置所对应的token_index
           
        new_start_positions = []
        for start in start_positions:
            start = offstart[0] if start < offstart[0] else start
            while start not in offstart:
                start -= 1
            new_start_positions.append(origin_offset2token_idx_start[start])
          

        new_end_positions = []
        for end in end_positions:
            end = offend[-1] if end > offend[-1] else end
            while end not in offend:
                end += 1
            new_end_positions.append(origin_offset2token_idx_end[end])        

        # 根据token索引为context创建label mask
        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()


        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        # 创建起始位置和结束位置的 label 列表
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # 截断超过最大长度的内容
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # 确保最后一个token必须是[SEP]
        sep_token = tokenizer.convert_tokens_to_ids("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0
        
        # 制作实体首尾位置匹配矩阵
        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
#             if start >= seq_len or end >= seq_len:
            if end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
        ]


def compute_loss(span_loss_candidates, start_logits, end_logits, span_logits,
                 start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
    batch_size, seq_len = start_logits.size()

    # view和reshape功能相同
    # batch=4 [4,512] => [2048]
    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    # [4,512] => [4,512,1] => [4,512,512]
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    # [4,512] => [4,1,512] => [4,512,512]
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    # 过滤行列中存在掩码内容，设置为False
    match_label_mask = match_label_row_mask & match_label_col_mask
    # mask矩阵下三角部分设置为False
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    if span_loss_candidates == "all":
        # naive mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    else:
        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if span_loss_candidates == "gold":
            # [batch,seq_len] => [batch,seq_len,1] => [batch,seq_len,seq_len]
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        elif span_loss_candidates == "pred_gold_random":
            gold_and_pred = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )

            data_generator = torch.Generator()
            data_generator.manual_seed(0)

            random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
            random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
            global device
            random_matrix = random_matrix.to(device)
            match_candidates = torch.logical_or(
                gold_and_pred, random_matrix
            )
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    
    # reduction='none' 不计算平均误差，返回shape和输入项一致
    bce_loss = BCEWithLogitsLoss(reduction='none')

    # 实体起始位置误差
    start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
    start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
    # 实体结束位置误差
    end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
    end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
    # 首位相交位置概率误差
    match_loss = bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
    match_loss = match_loss * float_match_label_mask
    match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

    return start_loss, end_loss, match_loss


def validation_step(model, batch):
    # 评估方法
    output = {}
    tokens, type_ids, start_labels, end_labels, \
    start_label_mask, end_label_mask, \
    match_labels = batch

    attention_mask = (tokens != 0).bool()
    start_logits, end_logits, span_logits = model(tokens, type_ids, attention_mask)

    start_loss, end_loss, match_loss = compute_loss('pred_and_gold', start_logits=start_logits,
                                                     end_logits=end_logits,
                                                     span_logits=span_logits,
                                                     start_labels=start_labels,
                                                     end_labels=end_labels,
                                                     match_labels=match_labels,
                                                     start_label_mask=start_label_mask,
                                                     end_label_mask=end_label_mask
                                                     )

    total_loss = start_loss + end_loss + match_loss

    output[f"val_loss"] = total_loss
    output[f"start_loss"] = start_loss
    output[f"end_loss"] = end_loss
    output[f"match_loss"] = match_loss

    # 取所有大于0的结果为预测结果
    start_preds, end_preds = start_logits > 0, end_logits > 0
    span_f1_stats = query_span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                  start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                  match_labels=match_labels)
    output["span_f1_stats"] = span_f1_stats
    return output

# 评估方法
def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # 起始位置预测 [bsz, seq_len]
    start_preds = start_preds.bool()
    # 结束位置预测 [bsz, seq_len]
    end_preds = end_preds.bool()
    # 把起始、结束位置预测与首尾位置预测合并
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    # 起始、结束真实位置合并后作为首尾位置预测的mask
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    # 保留mask矩阵的上三角
    match_label_mask = torch.triu(match_label_mask, 0)  # 起始位置小于等于结束位置
    match_preds = match_label_mask & match_preds
    
    # 被模型预测为正类的正样本 True Positive
    tp = (match_labels & match_preds).long().sum()
    # 被模型预测为正类的负样本 False Positives
    fp = (~match_labels & match_preds).long().sum()
    # 被模型预测为负类的正样本 False Negatives
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])
