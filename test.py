# label_ids 11
# input_ids = ntokens 12
# valid 12

# from transformers import AutoModel, AutoTokenizer, BertTokenizer
# model = AutoModel.from_pretrained('twmkn9/bert-base-uncased-squad2', mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
# ishan/bert-base-uncased-mnli
# twmkn9/bert-base-uncased-squad2
# deepset/bert-large-uncased-whole-word-masking-squad2
# tokenizer = BertTokenizer.from_pretrained("/home/whou/workspace/pretrained_models/bert-base-cased")
# print(tokenizer.tokenize("BRUSSELS"))
# from seqeval.metrics.sequence_labeling import get_entities
# entities = get_entities(['B-LOC', 'B-LOC', 'O', 'I-LOC'])
# print(entities)

tensor([  101,  1966,   117,  1117,  1494,  1111,  1103, 16752,  4487,  2612,
         1108,  2609,   117,  1122,  1108,  1536,  1106,  1782,  1140,  1696,
        23476,  1116,  1107,  1103, 11546,  3469,   117,  1867, 14895,   119,
         3938, 14809,  1183,   117,   170,  8472,  7319,  1150,  1173,  1462,
         1113,  1103,  3279,  7829,  2341,   119,   102,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0],
       device='cuda:0')

[ self.bert( input_ids[i:i+1],attention_mask=attention_mask[i:i+1],token_type_ids=torch.zeros_like(token_type_ids)[i:i+1]) for i in range(32) ]