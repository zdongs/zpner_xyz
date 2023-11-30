import jsonlines
import random
import pickle

def jsonl_pre(jsonl_path:str):

    sentences = []

    with jsonlines.open(jsonl_path) as reader:
        for d in reader:
            # 删除每句结尾的换行符
            text = d['text'].rstrip('\n')
            
            # 简易分词
            tk = list(text)
            
            tg = ['O'] * len(tk)
            for tag in d['label']:
                start = tag[0]
                end = tag[1]
                label = tag[2]
                
                # # 以下注释是当使用其他分词方式时可以采用的标签对齐方案（如jieba）
                # # 本脚本中的情况，直接对齐就可以
                #  start_idx = len(''.join(tk[:start]))
                #  end_idx = start_idx + len(tk[start:end])
                tg[start] = 'B-'+label
                for i in range(start+1, end):
                    tg[i] = 'I-'+label
                    
            sentences.append((tk, tg))
    
    # 拆分并保存 
    # 这个设计是让创建vocab时也可以一并把训练集和测试集分出来
    random.seed(2)
    random.shuffle(sentences)
    splitpoint = len(sentences) // 8
    train = sentences[splitpoint:]
    test = sentences[:splitpoint]
    with open("./data/bio_train.lt", "wb") as f:
        pickle.dump(train, f)
    with open("./data/bio_test.lt", "wb") as f:
        pickle.dump(test, f)

    return sentences

if __name__ == '__main__':


    path = './downloads/test.jsonl'
    sentences = jsonl_pre(path)

    # 单句检查
    for tk,tg in sentences:
        for a,b in zip(tk,tg):
            print(a,b)
        break

    # # 全部检查
    # print(sentences)
