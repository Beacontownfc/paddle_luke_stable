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
import paddle.nn.functional as F
import paddle

from luke_model.model import LukeEntityAwareAttentionModel


class LukeForEntityTyping(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super(LukeForEntityTyping, self).__init__(args.model_config)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        weight_attr, bias_attr = None, None
        if args.do_train:
            weight_attr = paddle.ParamAttr(name="weight", initializer=paddle.nn.initializer.Normal(mean=0.0, 
                                                std=args.model_config.initializer_range))
            bias_attr = paddle.ParamAttr(name="bias", initializer=paddle.nn.initializer.Constant(value=0.0))
        self.typing = nn.Linear(args.model_config.hidden_size, num_labels, weight_attr=weight_attr, bias_attr=bias_attr)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        labels=None,
    ):
        encoder_outputs = super(LukeForEntityTyping, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        feature_vector = encoder_outputs[1][:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.typing(feature_vector)
        if labels is None:
            return logits

        return (F.binary_cross_entropy_with_logits(logits.reshape([-1]), labels.reshape([-1]).astype('float32')),)