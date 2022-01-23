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
"""create squad train squad_data"""
import os
import json
from src.model_utils.config_args import args_config as args
from src.utils.utils import create_dir_not_exist
from utils.luke_tokenizer import LukeTokenizer
#from paddlenlp.transformers import RobertaT
from src.utils.entity_vocab import EntityVocab
from src.reading_comprehension.dataProcessing import build_data_change

args.wiki_link_db_file = args.wiki_data + "enwiki_20160305.pkl"
args.model_redirects_file = args.wiki_data + "enwiki_20181220_redirects.pkl"
args.link_redirects_file = args.wiki_data + "enwiki_20160305_redirects.pkl"

current_dir = os.path.dirname(os.path.abspath(__file__))
create_dir_not_exist(args.output_dir)

args.bert_model_name = 'roberta-large'
args.max_mention_length = 30
with open(current_dir + '/squad_data/metadata.json') as f:
    metadata = json.load(f)
entity_vocab = EntityVocab(current_dir + '/squad_data/entity_vocab.tsv')
args.entity_vocab = entity_vocab
args.tokenizer = LukeTokenizer(current_dir + '/squad_data/vocab.json', current_dir + '/squad_data/merges.txt')
build_data_change(args)
