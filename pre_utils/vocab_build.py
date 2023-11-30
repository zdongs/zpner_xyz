from torchtext.vocab import build_vocab_from_iterator
import pickle

def vocab_gen(senlist):
    # 制作词汇表
    gen = (line[0] for line in senlist)
    vocab = build_vocab_from_iterator(gen, specials=["<PAD>", "<UNK>"])
    # vocab.set_default_index(vocab["<UNK>"])

    # 保存词汇表
    with open('./data/vocab.bin', 'wb') as file:
        pickle.dump(vocab,file)
        print('词典已保存')

def t2i(senlist):
    tag2idx = {"<START>":0,"<STOP>":1,"<PAD>":2}
    for line in senlist:
        for k in line[1]:
            if k not in tag2idx:
                tag2idx[k] = len(tag2idx)
    with open('./data/tag2idx.bin', 'wb') as file:
        pickle.dump(tag2idx,file)
        print('t2i已保存')

if __name__ == '__main__':
    from jsonl_pre import jsonl_pre
    path = './downloads/all.jsonl'
    sentences = jsonl_pre(path)
    vocab_gen(sentences)
    t2i(sentences)
