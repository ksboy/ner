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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from metrics import f1_score_identification, precision_score_identification, recall_score_identification, \
    accuracy_score_entity_classification, accuracy_score_token_classification, \
    f1_score_token_classification, precision_score_token_classification, recall_score_token_classification 
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from model import BertForTokenClassificationMultiTask as AutoModelForTokenClassification
# from utils_ner_multi_task import convert_examples_to_features, read_examples_from_file
from utils_ee_multi_task import convert_examples_to_features, read_examples_from_file
# from utils import get_labels_ner as get_labels
from utils import get_labels_ee as get_labels

from utils import write_file
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels_i, labels_c, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #     os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    best_metric = 0 
    patience =0
    for _ in train_iterator:
        print(_, "epoch")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_i": batch[3],  "labels_c": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use token_type_ids

            outputs = model(**inputs)
            loss, loss_i, loss_c = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            logger.info("loss: %f, loss_i: %f, loss_c: %f", loss.item(), loss_i.item(), loss_c.item()) 
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels_i, labels_c, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    current_metric = results["f1"]
                    if current_metric <= best_metric:
                        if best_metric != 0: patience += 1
                        print("=" * 80)
                        print("Best Metric", best_metric)
                        print("Current Metric", current_metric)
                        print("=" * 80)
                        if patience > args.early_stop:
                            print("Out of patience !  Stop Training !")
                            return global_step, tr_loss / global_step
                    else:
                        patience = 0
                        print("=" * 80)
                        print("Best Metric", best_metric)
                        print("Current Metric", current_metric)
                        print("Saving Model......")
                        print("=" * 80)
                        best_metric = current_metric
                        
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels_i, labels_c, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels_i, labels_c, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # fix the bug when using mult-gpu during evaluating
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_i, preds_c = None, None
    out_label_ids_i, out_label_ids_c = None, None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_i": batch[3], "labels_c": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use token_type_ids
            outputs = model(**inputs)
            (tmp_eval_loss, tmp_eval_loss_i, tmp_eval_loss_c), (logits_i, logits_c) = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds_i is None:
            preds_i = logits_i.detach().cpu().numpy()
            out_label_ids_i = inputs["labels_i"].detach().cpu().numpy()
            preds_c = logits_c.detach().cpu().numpy()
            out_label_ids_c = inputs["labels_c"].detach().cpu().numpy()
        else:
            preds_i = np.append(preds_i, logits_i.detach().cpu().numpy(), axis=0)
            out_label_ids_i = np.append(out_label_ids_i, inputs["labels_i"].detach().cpu().numpy(), axis=0)
            preds_c = np.append(preds_c, logits_c.detach().cpu().numpy(), axis=0)
            out_label_ids_c = np.append(out_label_ids_c, inputs["labels_c"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds_i = np.argmax(preds_i, axis=2)
    preds_c = np.argmax(preds_c, axis=2)

    label_map_i = {i: label for i, label in enumerate(labels_i)}
    label_map_c = {i: label for i, label in enumerate(labels_c)}
    label_map_c[pad_token_label_id]= 'O'

    out_label_list = [[] for _ in range(out_label_ids_i.shape[0])]
    preds_list = [[] for _ in range(out_label_ids_i.shape[0])]

    # BIO标注，不包括实体类型
    out_label_list_i = [[] for _ in range(out_label_ids_i.shape[0])]
    preds_list_i = [[] for _ in range(out_label_ids_i.shape[0])]

    # 实体类型+{O}
    out_label_list_c = [[] for _ in range(out_label_ids_i.shape[0])]
    preds_list_c = [[] for _ in range(out_label_ids_i.shape[0])]

    for i in range(out_label_ids_i.shape[0]):
        for j in range(out_label_ids_i.shape[1]):
            if out_label_ids_i[i, j] != pad_token_label_id:
                # label
                label_i = label_map_i[out_label_ids_i[i][j]]
                label_c = label_map_c[out_label_ids_c[i][j]]
                if label_i =="O" or label_c=="O":
                    assert label_i==label_c
                    label = "O"
                else:
                    label = label_i + "-" + label_c
                out_label_list[i].append(label)
                out_label_list_i[i].append(label_i)
                out_label_list_c[i].append(label_c)
                # pred
                pred_i = label_map_i[preds_i[i][j]]
                pred_c = label_map_c[preds_c[i][j]]
                if pred_i =="O" or pred_c=="O":
                    pred = "O"
                else:
                    pred = pred_i + "-" + pred_c
                preds_list[i].append(pred)
                preds_list_i[i].append(pred_i)
                preds_list_c[i].append(pred_c)

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "precision_i": precision_score_identification(out_label_list, preds_list),
        "recall_i": recall_score_identification(out_label_list, preds_list),
        "f1_i": f1_score_identification(out_label_list, preds_list),
        "precision_c": precision_score_token_classification(out_label_list, preds_list),
        "recall_c": recall_score_token_classification(out_label_list, preds_list),
        "f1_c": f1_score_token_classification(out_label_list, preds_list),
        "accuracy_token_c": accuracy_score_token_classification(out_label_list, preds_list),
        "accuracy_entity_c": accuracy_score_entity_classification(out_label_list, preds_list)
    }

    report = classification_report(out_label_list, preds_list, digits=4)
    
    logger.info("***** Eval results %s *****", prefix)
    logger.info("\n%s", report)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels_i, labels_c, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode, task=args.task, dataset=args.dataset)
        features = convert_examples_to_features(
            examples,
            labels_i, labels_c,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        # if args.local_rank in [-1, 0] and not args.overwrite_cache:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids_i = torch.tensor([f.label_ids_i for f in features], dtype=torch.long)
    all_label_ids_c = torch.tensor([f.label_ids_c for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids_i, all_label_ids_c)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="The dataset name.",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        required=False,
        help="The task name.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--early_stop", default=4, type=int,
                        help="early stop when metric does not increases any more")
    parser.add_argument("--freeze",
                        action="store_true",
                        help="freeze embedding layer")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--loss", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # if args.dataset in ['lic', 'ccks']:
    #     from utils_ee_joint import convert_examples_to_features, read_examples_from_file
    #     from utils import get_labels_ee as get_labels
    # elif args.dataset in ['conll-2003', 'ontonotes-5.0', 'support-wnut-5shot']:
    #     from utils_ner_joint import convert_examples_to_features, read_examples_from_file
    #     from utils import get_labels_ner as get_labels

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # if args.dataset in ['lic', 'ccks']:
    #     from utils_ee_joint import convert_examples_to_features, read_examples_from_file
    #     from utils import get_labels_ee as get_labels
    # elif args.dataset in ['conll-2003', 'ontonotes-5.0', 'support-wnut-5shot']:
    #     from utils_ner_joint import convert_examples_to_features, read_examples_from_file
    #     from utils import get_labels_ner as get_labels

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels_i = get_labels(args.labels, task=args.task, mode="identification")
    labels_c = get_labels(args.labels, task=args.task, mode="classification") + ['O']
    print(labels_c, labels_i)

    num_labels_i = len(labels_i)
    num_labels_c = len(labels_c)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )
    ## update config
    config.num_labels_i=num_labels_i
    config.num_labels_c=num_labels_c
    config.id2label_c={str(i): label for i, label in enumerate(labels_c)}
    config.label2id_c={label: i for i, label in enumerate(labels_c)}
    config.id2label_i={str(i): label for i, label in enumerate(labels_i)}
    config.label2id_i={label: i for i, label in enumerate(labels_i)}
    config.loss = args.loss

    if os.path.exists(os.path.join(args.model_name_or_path, "pytorch_model.bin")):
        unexpected_keys = ['classifier.weight', 'classifier.bias']
        state_dict = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"), map_location="cpu")
        # for key in state_dict.keys():
        #     print(key)
        for key in unexpected_keys:
            state_dict.pop(key, None)
    else:
        state_dict = None

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        state_dict = state_dict,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.freeze:
        # for param in model.bert.bert.parameters():
        #     param.requires_grad = False
        for name, param in model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
            else:
                print(name)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels_i, labels_c, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels_i, labels_c, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        checkpoints = [os.path.join(args.output_dir, 'checkpoint-best')]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            model = AutoModelForTokenClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result, predictions = evaluate(args, model, tokenizer, labels_i, labels_c, pad_token_label_id, mode="dev", prefix=checkpoint)
            # Save results
            output_eval_results_file = os.path.join(checkpoint, "eval_results.txt")
            with open(output_eval_results_file, "w") as writer:
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))
            # Save predictions
            output_eval_predictions_file = os.path.join(checkpoint, "eval_predictions.json")
            results = []
            for prediction in predictions:
                results.append({'labels':prediction})
            write_file(results, output_eval_predictions_file)  

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels_i, labels_c, pad_token_label_id, mode="test")
        # Save results
        output_test_results_file = os.path.join(checkpoint, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(checkpoint, "test_predictions.json")
        results = []
        for prediction in predictions:
            results.append({'labels':prediction})
        write_file(results,output_test_predictions_file)      

    return results


if __name__ == "__main__":
    main()
