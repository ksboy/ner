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

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, id, words, labels):
        """Constructs a InputExample.
        Args:
            id: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids


## lic 格式
def trigger_process_bio_lic(input_file, is_predict=False):
    raise NotImplementedError
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "words":list(row["text"]), "labels":labels})
            continue
        for event in row["event_list"]:
            trigger = event["trigger"]
            event_type = event["event_type"]
            trigger_start_index = event["trigger_start_index"]
            labels[trigger_start_index]= "B-{}".format(event_type)
            for i in range(1, len(trigger)):
                labels[trigger_start_index+i]= "I-{}".format(event_type)
                # labels[trigger_start_index+i]= "I-{}".format("触发词")
        results.append({"id":row["id"], "words":list(row["text"]), "labels":labels})
    # write_file(results,output_file)
    return results

## ccks格式
def trigger_process_bio_ccks(input_file, is_predict=False):
    raise NotImplementedError
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["content"])
        if is_predict: 
            results.append({"id":row["id"], "words":list(row["content"]), "labels":labels})
            continue
        for event in row["events"]:
            event_type = event["type"]
            for mention in event["mentions"]:
                if mention["role"]=="trigger":
                    trigger = mention["word"]
                    trigger_start_index, trigger_end_index = mention["span"]
                    labels[trigger_start_index]= "B-{}".format(event_type)
                    for i in range(trigger_start_index+1, trigger_end_index):
                        labels[i]= "I-{}".format(event_type)
                        # labels[i]= "I-{}".format("触发词")
                    break
        results.append({"id":row["id"], "words":list(row["content"]), "labels":labels})
    # write_file(results,output_file)
    return results

## lic格式
def role_process_bio_lic(input_file, add_event_type_to_role=False, is_predict=False):
    raise NotImplementedError
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "words":list(row["text"]), "labels":labels})
            continue
        for event in row["event_list"]:
            # print(event)
            event_type = event["event_type"]
            for arg in event["arguments"]:
                role = arg['role']
                if add_event_type_to_role: role = event_type + '-' + role
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                labels[argument_start_index]= "B-{}".format(role)
                for i in range(1, len(argument)):
                    labels[argument_start_index+i]= "I-{}".format(role)
                # if arg['alias']!=[]: print(arg['alias'])
        results.append({"id":row["id"], "words":list(row["text"]), "labels":labels})
    # write_file(results,output_file)
    return results

## ccks格式
def role_process_bio_ccks(input_file, add_event_type_to_role=False, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["content"])
        if is_predict:
            results.append({"id":row["id"], "words":list(row["content"]), "labels":labels})
            continue
        for event in row["events"]:
            event_type = event["type"]
            for arg in event["mentions"]:
                role = arg['role']
                if role=="trigger": continue
                if add_event_type_to_role: role = event_type + '-' + role
                argument_start_index, argument_end_index = arg["span"]
                # labels[argument_start_index]= "B-{}".format(role)
                labels[argument_start_index]= "B"
                for i in range(argument_start_index+1, argument_end_index):
                    # labels[i]= "I-{}".format(role)
                    labels[i]= "I"
                # if arg['alias']!=[]: print(arg['alias'])
                results.append({"id":row["id"], "words":list(row["content"]), "labels":labels,})
    # write_file(results,output_file)
    return results

def read_examples_from_file(data_dir, mode, task='role', dataset="ccks"):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    if dataset=="ccks":
        if task=='trigger': items = trigger_process_bio_ccks(file_path)
        elif task=='role': items = role_process_bio_ccks(file_path, add_event_type_to_role=True)
    elif dataset=="lic":
        if task=='trigger': items = trigger_process_bio_lic(file_path)
        elif task=='role': items = role_process_bio_lic(file_path, add_event_type_to_role=True)
    return [InputExample(**item) for item in items]


def convert_examples_to_features(
    examples,
    label_list,
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
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # # Use the real label id for all token of the word
                # label_ids.extend([label_map[label]] * len(word_tokens))
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

                # # only use the first token of the word
                # tokens.append(word_tokens[0])
                # label_ids.append(label_map[label])

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

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
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s", example.id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_ids=label_ids)
        )
    return features
