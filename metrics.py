from seqeval.metrics.sequence_labeling import get_entities, f1_score, precision_score, recall_score, accuracy_score
import copy

def recall_score_i(y_true, y_pred):
    y_true = copy.deepcopy(y_true)
    y_pred = copy.deepcopy(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            label_true = y_true[i][j]
            y_true[i][j]= label_true[0]
            label_pred = y_pred[i][j]
            y_pred[i][j] = label_pred[0]
    return recall_score(y_true, y_pred)

def precision_score_i(y_true, y_pred):
    y_true = copy.deepcopy(y_true)
    y_pred = copy.deepcopy(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            label_true = y_true[i][j]
            y_true[i][j]= label_true[0]
            label_pred = y_pred[i][j]
            y_pred[i][j] = label_pred[0]
    return precision_score(y_true, y_pred)

def f1_score_i(y_true, y_pred):
    y_true = copy.deepcopy(y_true)
    y_pred = copy.deepcopy(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            label_true = y_true[i][j]
            y_true[i][j]= label_true[0]
            label_pred = y_pred[i][j]
            y_pred[i][j] = label_pred[0]
    return f1_score(y_true, y_pred)

def accuracy_score_token(y_true, y_pred):
    y_true = copy.deepcopy(y_true)
    y_pred = copy.deepcopy(y_pred)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            label_true = y_true[i][j]
            if len(label_true) >1:
                y_true[i][j]= label_true[2:]
            lael_pred = y_pred[i][j]
            if len(label_true) >1:
                y_pred[i][j]= lael_pred[2:]
    return accuracy_score(y_true, y_pred)

def accuracy_score_entity(y_true, y_pred):
    entities_true = []
    entities_pred = []
    _entities_true = []
    _entities_pred = []
    for i, (labels, preds) in enumerate(zip(y_true, y_pred)):
        for entity in get_entities(labels):
            entity = list(entity)
            entities_true.append([i]+ entity)
            _entities_true.append([i] + entity[1:])
        for entity in get_entities(preds):
            entity = list(entity)
            entities_pred.append([i]+ entity)
            _entities_pred.append([i] + entity[1:])
    nb_correct = sum(y_t == y_p for y_t, y_p in zip(entities_true, entities_true))
    nb_true = sum(y_t == y_p for y_t, y_p in zip(_entities_true, _entities_true))
    score = nb_correct / nb_true
    return score