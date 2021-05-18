import json

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")

def remove_duplication(alist):
    res = []
    for item in alist:
        if item not in res:
            res.append(item)
    return res


def get_labels_ner(path, mode="ner"):
    if mode=="identification":
        return ["B","I","O"]
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if mode=="classification":
            return labels # + ["O"]
        elif mode=='ner':
            _labels = ["O"]
            for label in labels:
                _labels.extend(["B-"+label, "I-"+label])
            return _labels
    else:
        if mode=="classification":
            return ["MISC", "PER", "ORG", "LOC"] # + ["O"]
        elif mode=='ner':
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def get_labels_ee(path="./data/event_schema.json", task='role', mode="ner", target_event_type='', add_event_type_to_role=True):
    
    if mode=="identification":
        return ["B", "I", "O"]

    if not path:
        if mode=='ner':
            return ["O", "B-ENTITY", "I-ENTITY"]
        else:
            return ["O"]

    elif task=='trigger':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            event_type = row["event_type"]
            if mode == "ner":
                labels.append("B-{}".format(event_type))
                labels.append("I-{}".format(event_type))
            elif mode == "classification":
                labels.append(event_type)
        return remove_duplication(labels)

    elif task=='role' and target_event_type=='':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            event_type = row["event_type"]
            for role in row["role_list"]:
                role_type = role['role'] if not add_event_type_to_role else event_type + '-' + role['role']
                if mode == "ner":
                    labels.append("B-{}".format(role_type))
                    labels.append("I-{}".format(role_type))
                elif mode == "classification":
                    labels.append(role_type)
        return remove_duplication(labels)
        
    # 特定类型事件 [TASK] 中的角色
    elif task=='role' and target_event_type!='':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            event_type = row["event_type"]
            if event_type!=target_event_type:
                continue
            for role in row["role_list"]:
                role_type = role['role'] if not add_event_type_to_role else event_type + '-' + role['role']
                if mode == "ner":
                    labels.append("B-{}".format(role_type))
                    labels.append("I-{}".format(role_type))
                elif mode == "classification":
                    labels.append(role_type)
        return remove_duplication(labels)