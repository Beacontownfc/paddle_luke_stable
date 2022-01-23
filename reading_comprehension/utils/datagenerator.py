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

from paddle.io import Dataset
import numpy as np

class DataGenerator(Dataset):
    def __init__(self, features, args):
        super(DataGenerator, self).__init__()
        self.args = args
        self.all_word_ids = [f['word_ids'] for f in features]
        self.all_word_segment_ids = [f['word_segment_ids'] for f in features]
        self.all_word_attention_mask = [f['word_attention_mask'] for f in features]
        self.all_entity_ids = [f['entity_ids'] for f in features]
        self.all_entity_position_ids = [f['entity_position_ids'] for f in features]
        self.all_entity_segment_ids = [f['entity_segment_ids'] for f in features]
        self.all_entity_attention_mask = [f['entity_attention_mask'] for f in features]
        self.all_start_positions = [f['start_positions'] for f in features] if not args.evaluate else None
        self.all_end_positions = [f['end_positions'] for f in features] if not args.evaluate else None
        self.all_example_index = [f['example_indices'] for f in features] if args.evaluate else None
    
    def __getitem__(self, item):
        word_ids = self.to_numpy(self.all_word_ids[item])
        word_segment_ids = self.to_numpy(self.all_word_segment_ids[item])
        word_attention_mask = self.to_numpy(self.all_word_attention_mask[item])
        entity_ids = self.to_numpy(self.all_entity_ids[item])
        entity_position_ids = self.to_numpy(self.all_entity_position_ids[item])
        entity_segment_ids = self.to_numpy(self.all_entity_segment_ids[item])
        entity_attention_mask = self.to_numpy(self.all_entity_attention_mask[item])
        start_positions = np.array(self.all_start_positions[item], dtype=np.int64) if not self.args.evaluate else 0
        end_positions = np.array(self.all_end_positions[item], dtype=np.int64) if not self.args.evaluate else 0
        example_index = self.all_example_index[item] if self.args.evaluate else 0
        return word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, \
                    entity_segment_ids , entity_attention_mask, start_positions, end_positions, example_index

    def __len__(self):
        return len(self.all_word_ids)

    def to_numpy(self, value):
        return np.array(value, np.int64)

class DataGeneratorV2(Dataset):
    def __init__(self, features, args):
        super(DataGeneratorV2, self).__init__()
        self.args = args
        self.all_word_ids = [f.word_ids for f in features]
        self.all_word_segment_ids = [f.word_segment_ids for f in features]
        self.all_word_attention_mask = [f.word_attention_mask for f in features]
        self.all_entity_ids = [f.entity_ids for f in features]
        self.all_entity_position_ids = [f.entity_position_ids for f in features]
        self.all_entity_segment_ids = [f.entity_segment_ids for f in features]
        self.all_entity_attention_mask = [f.entity_attention_mask for f in features]
        self.all_start_positions = [f.start_positions for f in features]
        self.all_end_positions = [f.end_positions for f in features]
        self.all_example_index = [f.example_index for f in features]
        self.all_unique_id = [f.unique_id for f in features]
    
    def __getitem__(self, item):
        word_ids = self.pad_sequence(self.all_word_ids[item], 1, self.args.max_seq_length)
        word_segment_ids = self.pad_sequence(self.all_word_segment_ids[item], 0, self.args.max_seq_length)
        word_attention_mask = self.pad_sequence(self.all_word_attention_mask[item], 0, self.args.max_seq_length)
        entity_ids = self.pad_sequence(self.all_entity_ids[item], 0, self.args.max_entity_length)
        entity_position_ids = self.pad_sequence(self.all_entity_position_ids[item], -1, self.args.max_entity_length, True)
        entity_segment_ids = self.pad_sequence(self.all_entity_segment_ids[item], 0, self.args.max_entity_length)
        entity_attention_mask = self.pad_sequence(self.all_entity_attention_mask[item], 0, self.args.max_entity_length)
        start_positions = np.array(self.all_start_positions[item], dtype=np.int64)
        end_positions = np.array(self.all_end_positions[item], dtype=np.int64)
        example_index = self.all_example_index[item]
        unique_id = self.all_unique_id[item]
        return word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids, \
                    entity_segment_ids , entity_attention_mask, start_positions, end_positions, \
                    example_index, unique_id

    def __len__(self):
        return len(self.all_word_ids)

    def pad_sequence(self, value, padding_value, max_len, flag=False):
        if flag:
            if len(value) > max_len:
                return np.array(value[:max_len], np.int64)
            else:
                res = value + [[padding_value] * len(value[0])] * (max_len - len(value))
                return np.array(res, np.int64)
        else:
            if len(value) > max_len:
                return np.array(value[:max_len], np.int64)
            else:
                return np.array(value + [padding_value] * (max_len - len(value)), np.int64)



