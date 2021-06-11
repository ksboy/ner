from seqeval.metrics.sequence_labeling import get_entities, f1_score, precision_score, recall_score, accuracy_score
from seqeval.metrics.sequence_labeling import classification_report

# from sklearn.metrics import f1_score, precision_recall, recall_score, accuracy_score
# from sklearn.metrics import classification_report

import copy

# identification
def identification_score(y_true, y_pred):
    y_true = copy.deepcopy(y_true)
    y_pred = copy.deepcopy(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            label_true = y_true[i][j]
            y_true[i][j]= label_true[0]
            label_pred = y_pred[i][j]
            y_pred[i][j] = label_pred[0]
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)

def recall_score_identification(y_true, y_pred):
    p, r, f = identification_score(y_true, y_pred)
    return r

def precision_score_identification(y_true, y_pred):
    p, r, f = identification_score(y_true, y_pred)
    return p

def f1_score_identification(y_true, y_pred):
    p, r, f = identification_score(y_true, y_pred)
    return f


# classification： token 分类正确的 acc
def accuracy_score_token_classification(y_true, y_pred, ignore_i=True, remove_o=True):
    # 忽略B和I的区别
    if ignore_i:
        y_true = copy.deepcopy(y_true)
        y_pred = copy.deepcopy(y_pred)
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                label_true = y_true[i][j]
                if len(label_true) >1:
                    y_true[i][j]= label_true[2:]
                label_pred = y_pred[i][j]
                if len(label_true) >1:
                    y_pred[i][j]= label_pred[2:]
    y_true = [item for sublist in y_true for item in sublist]
    y_pred = [item for sublist in y_pred for item in sublist]
    
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

# classification： 非O 类别 的 micro-F1
def token_classification_score(y_true, y_pred, ignore_i=True, remove_o=True):
    # 忽略B和I的区别
    if ignore_i:
        y_true = copy.deepcopy(y_true)
        y_pred = copy.deepcopy(y_pred)
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                label_true = y_true[i][j]
                if len(label_true) >1:
                    y_true[i][j]= label_true[2:]
                label_pred = y_pred[i][j]
                if len(label_true) >1:
                    y_pred[i][j]= label_pred[2:]
    
    y_true = [item for sublist in y_true for item in sublist]
    y_pred = [item for sublist in y_pred for item in sublist]
    
    # 删除 O    
    if remove_o:
        nb_correct  = 0
        for t, p in zip(y_true, y_pred):
            if t == p and t != 'O': nb_correct += 1
        nb_pred = len([item for item in y_pred if item != 'O'])
        nb_true = len([item for item in y_true if item != 'O'])

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
    # from sklearn.metrics import classification_report
    # print(classification_report(y_true, y_pred))
    # print(p, r, f1)
    return p, r, f1

def precision_score_token_classification(y_true, y_pred):
    p, r, f1 = token_classification_score(y_true, y_pred)
    return p

def recall_score_token_classification(y_true, y_pred):
    p, r, f1 = token_classification_score(y_true, y_pred)
    return r

def f1_score_token_classification(y_true, y_pred):
    p, r, f1 = token_classification_score(y_true, y_pred)
    return f1

# classification: 边界识别正确的条件下，entity 分类 的正确率
def accuracy_score_entity_classification(y_true, y_pred):
    # 只保留 B和I，忽略类别标签
    _y_true = copy.deepcopy(y_true)
    _y_pred = copy.deepcopy(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            label_true = y_true[i][j]
            if len(label_true) >1:
                _y_true[i][j]= label_true[0]
            label_pred = y_pred[i][j]
            if len(label_true) >1:
                _y_pred[i][j]= label_pred[0]
                
    entities_true = []
    entities_pred = []
    _entities_true = []
    _entities_pred = []
    for i, (labels, preds, _labels, _preds) in enumerate(zip(y_true, y_pred, _y_true, _y_pred)):
        for entity in get_entities(labels):
            entity = list(entity)
            entities_true.append([i]+ entity)
        for entity in get_entities(_labels):
            entity = list(entity)
            _entities_true.append([i] + entity)
        for entity in get_entities(preds):
            entity = list(entity)
            entities_pred.append([i]+ entity)
        for entity in get_entities(_preds):
            entity = list(entity)
            _entities_pred.append([i] + entity)
    
    nb_correct  = 0
    for entity in entities_true:
        if entity in entities_pred:
            nb_correct += 1
    
    nb_true  = 0
    for entity in _entities_true:
        if entity in _entities_pred:
            nb_true += 1
    
    score = nb_correct / nb_true if nb_true else 0
    return score

if __name__ == "__main__":
    y_true = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-LOC', 'I-LOC', 'O']]
    y_pred = [['O', 'B-MISC', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'O', 'O']]
    acc = accuracy_score_entity_classification(y_true, y_pred)
    f1 = f1_score_token(y_true, y_pred)
    print(acc, f1)
    
    # y_true = [['O', 'O', 'B', 'I', 'I', 'I', 'O'], ['B', 'I', 'O']]
    # result = get_entities(y_true)
    # print(result)
