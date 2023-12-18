# 模型本体（含训练评估用Dataset、损失计算、评估方法）

import torch
import train_utils
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
                

# 单句推理
# 单句文本转换成四个类别字典（单句推理的数据预处理函数）

nice_tag = ['程度', '素质', '专业', '福利']

def gen_dicts(text):
    dicts = []     
    for tl in nice_tag:
        dic = {}
        dic["context"] = text
        dic["query"] = tl
        dicts.append(dic)
    return dicts

def pred_one(text,tokenizer):
    all_list = []
    dicts = gen_dicts(text)
    for data in dicts:
        query = data["query"]  # 问题描述
        context = data["context"]  # 内容描述
        
        query_context_tokens = tokenizer(query, context, return_offsets_mapping=True)
        tokens = query_context_tokens.input_ids
        type_ids = query_context_tokens.token_type_ids
        offsets = query_context_tokens.offset_mapping

        # 根据token索引为context创建label mask
        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        assert len(label_mask) == len(tokens)


        one_query_list =  [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask)]
        
        all_list.append(one_query_list)
    
    return all_list

# 实体预测矩阵生成、实体抽取、适用于单句文本推理的预处理（填充）函数

def query_span_f1(start_logits, end_logits, match_logits, start_label_mask, end_label_mask):
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    
    _, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    start_preds = start_logits > 0
    end_preds = end_logits > 0
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
    
    return match_preds

def extract_entities(text, match_preds, tokenizer):
    # 将 match_preds 转换为 NumPy 数组
    match_preds_np = match_preds.cpu().numpy()
    
    # 不添加[CLS]和[SEP]
    encoded_text = tokenizer.encode(text, add_special_tokens=False)
    # 获取 batch size 和序列长度
    bsz, seq_len, _ = match_preds_np.shape

    # 存储提取的实体
    entities = []

    # 遍历每个样本
    for i in range(bsz):
        query = nice_tag[i]
        entities_for_query = []

        # 遍历每个位置
        for j in range(seq_len):
            for k in range(j, seq_len):
                # 如果 match_preds 为 True，表示在当前位置存在标注
                if match_preds_np[i, j, k]:
                    # 提取文本片段并存储
                    # 减4是因为match_pred里面每行每列的前4个token是[CLS]标签名（占2个token）[SEP]
                    entity_text = encoded_text[j-4:k+1-4] # ???为什么结尾还要加1？
                    entity_text = tokenizer.decode(entity_text)
                    entity_text = ''.join([token for token in entity_text if token != ' '])
                    entities_for_query.append(entity_text)

        entities.append({query:entities_for_query})

    return entities

def test_collate_fn(batch):
    filled_tensors = [
        pad_sequence(samples, batch_first=True, padding_value=values) 
        for samples,values in zip(zip(*batch),[0,1,0,0])
    ]
    
    return filled_tensors

# 单句推理主流程

def call_entities(one_text, model, tokenizer):
    
    one_list = pred_one(one_text, tokenizer)

    one_dl = DataLoader(one_list, batch_size=4, collate_fn=test_collate_fn)

    model.to("cuda")
    model.eval()

    for batch in one_dl:
        batch = (item.to("cuda") for item in batch)
        tokens, type_ids, start_label_mask, end_label_mask = batch
        attention_mask = (tokens != 0).bool()
        with torch.no_grad():
            start_logits, end_logits, span_logits = model(tokens, type_ids, attention_mask)
            match_preds = query_span_f1(start_logits, end_logits, span_logits, start_label_mask, end_label_mask)

        entities = extract_entities(one_text, match_preds, tokenizer)
        return entities

def pre_model_tokenizer(load_model = "model/mrc-ner_f1_0.627926.pth"):

    bert_model = 'bert_model/chinese-roberta-wwm-ext-large'

    # 加载模型

    checkpoint = torch.load(load_model)

    #=========在DDP中保存的model需要去掉model包装=====
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if 'module' in key:
            new_key = key[7:]  # 去除'module.'前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    #==============================================
        
    bert_config = train_utils.BertQueryNerConfig.from_pretrained(bert_model)
    model = train_utils.BertQueryNER.from_pretrained(bert_model, config = bert_config)

    model.load_state_dict(new_state_dict)
    tokenizer = BertTokenizerFast.from_pretrained(bert_model)

    return model, tokenizer

if __name__ == "__main__":
    
    model, tokenizer = pre_model_tokenizer()
    
    text1 = "任职要求：1、计算机、通信、电子、数学等相关专业，本科及以上学历，3年以上开发经验； 2、良好的工程能力，能独立将算法模型应用于实际产品研发；1、熟悉基础的数据结构和算法2、有一定的机器学习技术和深度学习技术基础3、熟悉python/c++/go等一种或多种编程语言4、有学习热情，有上进心"
            
    entities = call_entities(text1, model, tokenizer)
    print(entities)


