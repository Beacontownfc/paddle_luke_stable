#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from transformers import RobertaTokenizer
import argparse

import paddle
from paddle.io import DataLoader
from tqdm import tqdm
from luke_model.entity_vocab import EntityVocab, MASK_TOKEN
from luke_model.model import LukeConfig
from trainer import Trainer
from open_entity import LukeForEntityTyping
from utils import ENTITY_TOKEN, convert_examples_to_features, DatasetProcessor
from datagenerator import DataGenerator
from word_tokenizer import AutoTokenizer
import pickle
import numpy as np
import random


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--checkpoint_file", type=str, default='luke.pt')
parser.add_argument("--data_dir", type=str, default='data/')
parser.add_argument("--do_eval", type=bool, default=True)
parser.add_argument("--do_train", type=bool, default=True)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=12)
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--entity_vocab_size", default=500000)
parser.add_argument("--vocab_size", default=50265)
parser.add_argument("--entity_emb_size", default=256)
parser.add_argument("--pretrain_model", default='/home/aistudio/data/data123707/paddle_luke.pt')
parser.add_argument("--output_dir", default="output/")
parser.add_argument("--max_answer_length", default=30)
parser.add_argument("--max_entity_length", default=128)
parser.add_argument("--max_query_length", default=64)
parser.add_argument("--max_seq_length", default=512)
parser.add_argument("--tokenizer", default=None)
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--max_mention_length", default=30)
parser.add_argument("--gradient_accumulation_steps", default=2)
parser.add_argument("--warmup_proportion", type=float, default=0.09)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--adam_b1", type=float, default=0.9)
parser.add_argument("--adam_b2", type=float, default=0.98)
parser.add_argument("--weight_decay", type=float, default=0.01)
args = parser.parse_args()

abs_path = os.path.abspath(__file__)
abs_path_dir = '/'.join(abs_path.split('/')[:-1])

args.entity_vocab = EntityVocab(abs_path_dir + 'entity_vocab.tsv')
args.tokenizer = RobertaTokenizer(abs_path_dir + '/data/vocab.json', abs_path_dir + '/data/merges.txt')

def change_state_dict(state_dict, config):
    not_transpose_list = ['embeddings.word_embeddings.weight',
                          'embeddings.position_embeddings.weight',
                          'embeddings.token_type_embeddings.weight',
                          'entity_embeddings.entity_embeddings.weight',
                          'entity_embeddings.position_embeddings.weight',
                          'entity_embeddings.token_type_embeddings.weight']
    new_state_dict = state_dict.copy()
    for k, v in new_state_dict.items():
        if k not in not_transpose_list and 'LayerNorm' not in k:
            new_state_dict[k] = paddle.t(v)
    for num in range(config.num_hidden_layers):
        for attr_name in ("weight", "bias"):
            if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in state_dict:
                new_state_dict[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = state_dict[
                    f"encoder.layer.{num}.attention.self.query.{attr_name}"
                ]
            if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in state_dict:
                new_state_dict[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = state_dict[
                    f"encoder.layer.{num}.attention.self.query.{attr_name}"
                ]
            if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in state_dict:
                    new_state_dict[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = state_dict[
                        f"encoder.layer.{num}.attention.self.query.{attr_name}"
                ]

    return new_state_dict

def check_state_dict(cur_model_state_dict, trg_model_state_dict):
    def mul(shape):
        s = 1
        for e in shape:
            s *= e
        return s

    for k, v in cur_model_state_dict.items():
        x = trg_model_state_dict.get(k, False)
        if isinstance(x, bool):
            print(k + " 参数未加载!!!")
        else:
            if x.shape != v.shape or (v == x).sum() != mul(v.shape):
                print(k + ' 参数未加载!!!')

def run(args):
    # 设置种子
    paddle.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.model_config = LukeConfig(vocab_size=args.vocab_size,
                                   entity_vocab_size=args.entity_vocab_size,
                                   bert_model_name='roberta-large',
                                   entity_emb_size=args.entity_emb_size)

    args.model_config.vocab_size += 1
    args.model_weights = paddle.load(args.pretrain_model)
    word_emb = args.model_weights["embeddings.word_embeddings.weight"]
    marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    args.model_weights["embeddings.word_embeddings.weight"] = paddle.concat([word_emb, marker_emb])
    args.tokenizer.add_special_tokens(dict(additional_special_tokens=[ENTITY_TOKEN]))

    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_config.entity_vocab_size = 2
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = paddle.concat([entity_emb[:1], mask_emb])

    train_dataloader, _, features, _ = load_examples(args, fold="train")
    num_labels = len(features[0].labels)

    results = {}

    if args.do_train:
        model = LukeForEntityTyping(args, num_labels)
        args.model_weights = change_state_dict(args.model_weights, args.model_config)
        model.set_state_dict(args.model_weights)
        check_state_dict(model.state_dict(), args.model_weights)
        del args.model_weights

        def step_callback(model, global_step, max_f1, output_dir):
            """训练时开启验证"""
            if (global_step + 1) % num_train_steps_per_epoch == 0:
                dev_results = evaluate(args, model, fold="dev")
                if max_f1 < dev_results['f1']:
                    max_f1 = dev_results['f1']
                    paddle.save(model.state_dict(), os.path.join(output_dir, 'luke.pt'))
            return max_f1

        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)


        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps, step_callback=step_callback
        )
        trainer.train()


    if args.do_eval:
        args.do_train = False
        model = LukeForEntityTyping(args, num_labels)
        model.set_state_dict(paddle.load(args.output_dir + args.checkpoint_file))

        for eval_set in ["test"]:
            output_file = os.path.join(args.output_dir, f"{eval_set}_predictions.jsonl")
            results.update({f"{eval_set}_{k}": v for k, v in evaluate(args, model, eval_set, output_file).items()})

    print("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results


def evaluate(args, model, fold="dev", output_file=None):
    """评估模型 模型评估代码来自官方提供"""
    dataloader, _, _, label_list = load_examples(args, fold=fold)
    model.eval()

    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc=fold):  
        with paddle.no_grad():
            logits = model( word_ids=batch[0],
                            word_segment_ids=batch[1],
                            word_attention_mask=batch[2],
                            entity_ids=batch[3],
                            entity_position_ids=batch[4],
                            entity_segment_ids=batch[5],
                            entity_attention_mask=batch[6],
                            labels=None)
    
        logits = logits.tolist()
        labels = batch[7].tolist()

        all_logits.extend(logits)
        all_labels.extend(labels)

    all_predicted_indexes = []
    all_label_indexes = []
    for logits, labels in zip(all_logits, all_labels):
        all_predicted_indexes.append([i for i, v in enumerate(logits) if v > -0.15])  # 决策边界设为-0.15
        all_label_indexes.append([i for i, v in enumerate(labels) if v > 0])

    if output_file:
        with open(output_file, "w") as f:
            for predicted_indexes, label_indexes in zip(all_predicted_indexes, all_label_indexes):
                data = dict(
                    predictions=[label_list[ind] for ind in predicted_indexes],
                    labels=[label_list[ind] for ind in label_indexes],
                )
                f.write(json.dumps(data) + "\n")

    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for predicted_indexes, label_indexes in zip(all_predicted_indexes, all_label_indexes):
        num_predicted_labels += len(predicted_indexes)
        num_gold_labels += len(label_indexes)
        num_correct_labels += len(frozenset(predicted_indexes).intersection(frozenset(label_indexes)))

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0

    recall = num_correct_labels / num_gold_labels
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(precision=precision, recall=recall, f1=f1)



def load_examples(args, fold="train"):

    processor = DatasetProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    logger.info("Creating features from the dataset...")

    features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length)


    data_generator = DataGenerator(features, args)
    def collate_fn(batch):
        def create_padded_sequence(k, padding_value):
            new_data = []
            max_len = 0
            for each_batch in batch:
                if len(each_batch[k]) > max_len:
                    max_len = len(each_batch[k])
            for each_batch in batch:
                new_data.append(each_batch[k] + [padding_value] * (max_len - len(each_batch[k])))
            return np.array(new_data, dtype='int64')
        
        return (
            create_padded_sequence(0, 1),
            create_padded_sequence(1, 0),
            create_padded_sequence(2, 0),
            create_padded_sequence(3, 0),
            create_padded_sequence(4, 0),
            create_padded_sequence(5, 0),
            create_padded_sequence(6, 0),
            create_padded_sequence(7, 0),
        )

    if fold in ("dev", "test"):
        dataloader = DataLoader(data_generator, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(data_generator, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, label_list

if __name__ == '__main__':
    paddle.set_device(args.device)
    run(args)