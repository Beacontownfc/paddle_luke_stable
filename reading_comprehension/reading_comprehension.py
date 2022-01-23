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

import paddle.nn as nn
from paddle.nn import CrossEntropyLoss
import paddle
from luke_model.model import LukeEntityAwareAttentionModel

class LukeForReadingComprehension(LukeEntityAwareAttentionModel):
    def __init__(self, args):
        super(LukeForReadingComprehension, self).__init__(args.model_config)
        weight_attr, bias_attr = None, None
        if args.do_train:
            weight_attr = paddle.ParamAttr(name="weight", initializer=paddle.nn.initializer.Normal(mean=0.0, 
                                                std=args.model_config.initializer_range))
            bias_attr = paddle.ParamAttr(name="bias", initializer=paddle.nn.initializer.Constant(value=0.0))
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2, weight_attr=weight_attr, bias_attr=bias_attr)
        #self.apply(self.init_weights)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        start_positions=None,
        end_positions=None,
    ):
        encoder_outputs = super(LukeForReadingComprehension, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask)

        word_hidden_states = encoder_outputs[0][:, : word_ids.shape[1], :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = paddle.split(logits, 2, -1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,)
        else:
            outputs = tuple()

        return outputs + (start_logits, end_logits,)

