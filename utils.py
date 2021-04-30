import json
def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")

def get_labels(path, mode="ner"):
    if mode=="identification":
        return ["B","I","O"]
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if mode=="classification":
            return labels + ["O"]
        else:
            _labels = ["O"]
            for label in labels:
                _labels.extend(["B-"+label, "I-"+label])
            return _labels
    else:
        if mode=="classification":
            return ["MISC", "PER", "ORG", "LOC", "O"]
        elif mode=="identification":
            return ["B","I","O"]
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
