import json
import logging
import multiprocessing
import os
import joblib
import numpy as np
import random
import paddle
import pickle
from paddle.io import DataLoader, RandomSampler, DistributedBatchSampler
from tqdm import tqdm
from luke_model.model import LukeConfig
from utils.trainer import Trainer
from reading_comprehension import LukeForReadingComprehension
from utils.datagenerator import DataGenerator, DataGeneratorV2
import argparse
import collections
from src.reading_comprehension.squad_get_predictions import *
from src.reading_comprehension.squad_postprocess import *
import paddle.distributed as dist
from utils.luke_tokenizer import LukeTokenizer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dump_db_file", type=str, default='')
parser.add_argument("--mention_db_file", type=str)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--pool_size", default=multiprocessing.cpu_count())
parser.add_argument("--chunk_size", default=100)
parser.add_argument("--compress", default=3)
parser.add_argument("--checkpoint_file", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--do_eval", type=str, default='')
parser.add_argument("--do_train", type=str, default='')
parser.add_argument("--doc_stride", default=128)
parser.add_argument("--eval_batch_size", default=32)
parser.add_argument("--link_redirects_file", type=str, default="")
parser.add_argument("--max_answer_length", default=30)
parser.add_argument("--max_entity_length", default=128)
parser.add_argument("--max_query_length", default=64)
parser.add_argument("--max_seq_length", default=512)
parser.add_argument("--min_mention_link_prob", default=0.01)
parser.add_argument("--model_redirects_file", type=str, default="")
parser.add_argument("--n_best_size", default=20)
parser.add_argument("--no_entity", default=False)
parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
parser.add_argument("--num_train_epochs", default=2)
parser.add_argument("--seed", default=42)
parser.add_argument("--train_batch_size", default=8)
parser.add_argument("--wiki_link_db_file", type=str, default="")
parser.add_argument("--with_negative", default=False)
parser.add_argument("--max_mention_length", default=30)
parser.add_argument("--entity_vocab_size", default=500000)
parser.add_argument("--vocab_size", default=50265)
parser.add_argument("--entity_emb_size", default=256)
parser.add_argument("--device", default='gpu')
parser.add_argument("--gradient_accumulation_steps", default=3)
parser.add_argument("--pretrain_model", type=str, default='')
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_proportion", type=float, default=0.06)
parser.add_argument("--learning_rate", type=float, default=20e-6)
parser.add_argument("--adam_b1", type=float, default=0.9)
parser.add_argument("--adam_b2", type=float, default=0.99)
parser.add_argument("--multi_cards", type=str, default='')

args = parser.parse_args()
current_dir = os.path.dirname(os.path.abspath(__file__))
args.tokenizer = LukeTokenizer(current_dir + '/squad_data/vocab.json', current_dir + '/squad_data/merges.txt')


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
    paddle.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.model_config = LukeConfig(vocab_size=args.vocab_size,
                                   entity_vocab_size=args.entity_vocab_size,
                                   bert_model_name='roberta-large',
                                   entity_emb_size=args.entity_emb_size)

    if args.do_train:
        model = LukeForReadingComprehension(args)
        model.entity_embeddings.entity_embeddings.weight.stop_gradient = True

        print('加载预训练模型......')
        state_dict = paddle.load(args.pretrain_model)
        state_dict = change_state_dict(state_dict, args.model_config)
        model.set_state_dict(state_dict)
        check_state_dict(model.state_dict(), state_dict)
        if args.multi_cards:
            dist.init_parallel_env()
            model = paddle.DataParallel(model)
        train_dataloader, _, _, _ = load_examples(args, evaluate=False)

        num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps)
        trainer.train()

    if args.do_train:
        paddle.save(model.state_dict(), os.path.join(args.output_dir, args.checkpoint_file))

    if args.do_eval:
        args.do_train = False
        model = LukeForReadingComprehension(args)

        model.set_state_dict(paddle.load(os.path.join(args.output_dir, args.checkpoint_file)))
        evaluate(args, model, prefix="")


def evaluate(args, model, prefix=""):
    dataloader, examples, features, processor = load_examples(args, evaluate=True)
    all_results = []
    print('开始评估模型......')
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    for batch in tqdm(dataloader, desc="eval"):
        model.eval()
        # inputs = {k: v.to(args.device) for k, v in batch.items() if k != "example_indices"}
        with paddle.no_grad():
            outputs = model(word_ids=batch[0],
                            word_segment_ids=batch[1],
                            word_attention_mask=batch[2],
                            entity_ids=batch[3],
                            entity_position_ids=batch[4],
                            entity_segment_ids=batch[5],
                            entity_attention_mask=batch[6])

        for i, example_index in enumerate(batch[-1]):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits, end_logits = [o[i] for o in outputs]
            start_logits = start_logits.tolist()
            end_logits = end_logits.tolist()
            all_results.append(RawResult(unique_id, start_logits, end_logits))
    all_predictions = write_predictions(args, examples, features, all_results, 20, 30, False)
    SQuad_postprocess(os.path.join(current_dir + '/squad_data/data/', processor.dev_file), all_predictions, output_metrics="output.json")


def load_examples(args, evaluate=False):
    args.evaluate = evaluate
    features = []
    if not evaluate:
        print('从JSON中加载训练数据集......')
        data_file = args.data_dir + 'train.json'
    else:
        print('从JSON中加载测试数据集......')
        data_file = args.data_dir + 'eval_data.json'
    with open(data_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            features.append(json.loads(line))
            line = f.readline()

    if evaluate:
        data_generator = DataGenerator(features, args)
        dataloader = DataLoader(data_generator, batch_size=args.eval_batch_size)
        with open(args.data_dir + 'eval_obj.pickle', 'rb') as f:
            eval_obj = pickle.load(f)
        examples, features, processor = eval_obj.examples, eval_obj.features, eval_obj.processor
    else:
        data_generator = DataGenerator(features, args)
        if args.multi_cards:
            batch_sampler = DistributedBatchSampler(data_generator,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=True,
                                                    drop_last=False)
            dataloader = DataLoader(data_generator, batch_sampler=batch_sampler)
        else:
            dataloader = DataLoader(data_generator, batch_size=args.train_batch_size, shuffle=True)
        examples, features, processor = None, None, None

    return dataloader, examples, features, processor


if __name__ == '__main__':
    paddle.set_device(args.device)
    run(args)
