from torch import long, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def mydl(d_train,d_test,batch_size,vocab,tag2idx):

    def collate(batch_data):
        token_idx, label = [],[]
        for tk,tg in batch_data:
            index = vocab(tk)
            token_idx.append(tensor(index,dtype=long))
            label.append(tensor([tag2idx[t] for t in tg],dtype=long))
        x = pad_sequence(token_idx,batch_first=True)
        y = pad_sequence(label,True,tag2idx['<PAD>'])
        return x,y

    return DataLoader(d_train,batch_size,True,collate_fn=collate),DataLoader(d_test,batch_size,collate_fn=collate)

if __name__ == "__main__":
    import pickle
    with open("./data/bio_train.lt", "rb") as f:
        d_train = pickle.load(f)
    with open("./data/bio_test.lt", "rb") as f:
        d_test = pickle.load(f)

    vocab_list = []
    # 加载词汇表
    with open('./data/vocab.bin', 'rb') as file:
        vocab = pickle.load(file)

    vocab.set_default_index(vocab['<UNK>'])  # 重新设置未知词索引

    with open('./data/tag2idx.bin', 'rb') as file:
        tag2idx = pickle.load(file)

    dl_train,dl_test = mydl(d_train,d_test,1,vocab,tag2idx)

    for a in dl_train:
        print(a)

