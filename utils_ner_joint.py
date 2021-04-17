# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import json
from seqeval.metrics.sequence_labeling import get_entities
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, segment_ids, labels_i, labels_c):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.segment_ids = segment_ids # 目标trigger位置
        self.labels_i = labels_i
        self.labels_c = labels_c


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids_i, label_ids_c):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids_i = label_ids_i
        self.label_ids_c = label_ids_c


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels_i, labels_c = [], []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    if mode!='train':
                        segment_ids = [0] * len(words)
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, segment_ids=segment_ids, labels_i=labels_i, labels_c = labels_c))
                    else:
                        entities = get_entities(labels_i)
                        for _, start, end in entities:
                            segment_ids = [0] * len(words)
                            _labels_c = ['O']*len(words)
                            for i in range(start, end+1):
                                segment_ids[i] = 1
                                _labels_c[i] = labels_c[i]
                            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, segment_ids=segment_ids, labels_i=labels_i, labels_c = _labels_c))
                    guid_index += 1
                    words = []
                    labels_i, labels_c = [], []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    label = splits[-1].replace("\n", "")
                    if len(label)==1:
                        labels_i.append("O")
                        labels_c.append("O")
                    else:
                        labels_i.append(label.split("-")[0])
                        labels_c.append(label.split("-")[1])
                else:
                    # Examples could have no label for mode = "test"
                    labels_i.append("O")
                    labels_c.append("O")

        if words:
            if mode!='train':
                segment_ids = [0] * len(words)
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, segment_ids=segment_ids, labels_i=labels_i, labels_c = labels_c))
            else:
                entities = get_entities(labels_i)
                for _, start, end in entities:
                    segment_ids = [0] * len(words)
                    _labels_c = ['O']*len(words)
                    for i in range(start, end+1):
                        segment_ids[i] = 1
                        _labels_c[i] = labels_c[i]
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, segment_ids=segment_ids, labels_i=labels_i, labels_c = _labels_c))
    return examples


def convert_examples_to_features(
    examples,
    label_list_i,
    label_list_c,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    trigger_token_segment_id = 1,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map_i = {label: i for i, label in enumerate(label_list_i)}
    label_map_c = {label: i for i, label in enumerate(label_list_c)}
    label_map_c['O']= pad_token_label_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        segment_ids= []
        label_ids_i = []
        label_ids_c = []
        for word, segment_id, label_i, label_c in zip(example.words, example.segment_ids, example.labels_i, example.labels_c):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                segment_ids.extend([sequence_a_segment_id if not segment_id else trigger_token_segment_id]+ [sequence_a_segment_id] * (len(word_tokens) - 1))
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids_i.extend([label_map_i[label_i]] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_ids_c.extend([label_map_c[label_c]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids_i = label_ids_i[: (max_seq_length - special_tokens_count)]
            label_ids_c = label_ids_c[: (max_seq_length - special_tokens_count)]
            segment_ids = segment_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids_i += [pad_token_label_id]
        label_ids_c += [pad_token_label_id]
        segment_ids += [sequence_a_segment_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids_i += [pad_token_label_id]
            label_ids_c += [pad_token_label_id]
            segment_ids += [sequence_a_segment_id]

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids_i += [pad_token_label_id]
            label_ids_c += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids_i = [pad_token_label_id] + label_ids_i
            label_ids_c = [pad_token_label_id] + label_ids_c
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids_i = ([pad_token_label_id] * padding_length) + label_ids_i
            label_ids_c = ([pad_token_label_id] * padding_length) + label_ids_c
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids_i += [pad_token_label_id] * padding_length
            label_ids_c += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids_i) == max_seq_length
        assert len(label_ids_c) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids_i: %s", " ".join([str(x) for x in label_ids_i]))
            logger.info("label_ids_c: %s", " ".join([str(x) for x in label_ids_c]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, \
                label_ids_i=label_ids_i, label_ids_c=label_ids_c)
        )
    return features

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")

def get_labels(path, mode="classification"):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        if mode=="classification":
            return ["MISC", "PER", "ORG", "LOC"]
        elif mode=="identification":
            return ["B","I","O"]