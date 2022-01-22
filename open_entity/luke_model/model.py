import paddle.nn as nn
import paddle
from luke_model.utils.bert_model_utils import BertConfig, BertSelfOutput, \
    BertIntermediate, BertOutput, \
    BertPooler, BertEmbeddings, RobertaEmbeddings

#from paddlenlp.transformers.bert.modeling import BertPooler
import math
import paddle.nn.functional as F

class LukeConfig(BertConfig):
    def __init__(
        self, vocab_size: int, entity_vocab_size: int, bert_model_name: str, entity_emb_size: int = None, **kwargs
    ):
        super(LukeConfig, self).__init__(vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.bert_model_name = bert_model_name
        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size

class EntityEmbeddings(nn.Layer):
    def __init__(self, config: LukeConfig):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias_attr=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, entity_ids, position_ids, token_type_ids = None
    ):
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(paddle.clip(position_ids, min=0))
        position_embedding_mask = (position_ids != -1).astype('float32').unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = paddle.sum(position_embeddings, axis=-2)
        position_embeddings = position_embeddings / paddle.clip(position_embedding_mask.sum(axis=-2), min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class LukeModel(nn.Layer):
    def __init__(self, config: LukeConfig):
        super(LukeModel, self).__init__()

        self.config = config

        self.encoder = None
        self.pooler = BertPooler(config)

        self.embeddings = RobertaEmbeddings(config)
        self.embeddings.token_type_embeddings.stop_gradient = True
        self.entity_embeddings = EntityEmbeddings(config)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids=None,
        entity_position_ids=None,
        entity_segment_ids=None,
        entity_attention_mask=None,
    ):
        word_seq_size = word_ids.shape[1]

        embedding_output = self.embeddings(word_ids, word_segment_ids)

        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        if entity_ids is not None:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
            embedding_output = paddle.concat([embedding_output, entity_embedding_output], axis=1)

        encoder_outputs = self.encoder(embedding_output, attention_mask, [None] * self.config.num_hidden_layers)
        sequence_output = encoder_outputs[0]
        word_sequence_output = sequence_output[:, :word_seq_size, :]
        pooled_output = self.pooler(sequence_output)

        if entity_ids is not None:
            entity_sequence_output = sequence_output[:, word_seq_size:, :]
            return (word_sequence_output, entity_sequence_output, pooled_output,) + encoder_outputs[1:]
        else:
            return (word_sequence_output, pooled_output,) + encoder_outputs[1:]



    def _compute_extended_attention_mask(self, word_attention_mask, entity_attention_mask):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = paddle.concat([attention_mask, entity_attention_mask], axis=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.astype('float32')
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

class LukeEntityAwareAttentionModel(LukeModel):
    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__(config)
        self.config = config
        self.encoder = EntityAwareEncoder(config)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
    ):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = self._compute_extended_attention_mask(word_attention_mask, entity_attention_mask)

        return self.encoder(word_embeddings, entity_embeddings, attention_mask)



class EntityAwareSelfAttention(nn.Layer):
    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        return paddle.transpose(x.reshape(new_x_shape), perm=(0, 2, 1, 3))

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_size = word_hidden_states.shape[1]

        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(self.key(paddle.concat([word_hidden_states, entity_hidden_states], axis=1)))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = paddle.matmul(w2w_query_layer, paddle.transpose(w2w_key_layer, perm=(0, 1, 3, 2)))
        w2e_attention_scores = paddle.matmul(w2e_query_layer, paddle.transpose(w2e_key_layer, perm=(0, 1, 3, 2)))
        e2w_attention_scores = paddle.matmul(e2w_query_layer, paddle.transpose(e2w_key_layer, perm=(0, 1, 3, 2)))
        e2e_attention_scores = paddle.matmul(e2e_query_layer, paddle.transpose(e2e_key_layer, perm=(0, 1, 3, 2)))

        word_attention_scores = paddle.concat([w2w_attention_scores, w2e_attention_scores], axis=3)
        entity_attention_scores = paddle.concat([e2w_attention_scores, e2e_attention_scores], axis=3)
        attention_scores = paddle.concat([word_attention_scores, entity_attention_scores], axis=2)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(paddle.concat([word_hidden_states, entity_hidden_states], axis=1))
        )
        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = paddle.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class EntityAwareAttention(nn.Layer):
    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = paddle.concat([word_hidden_states, entity_hidden_states], axis=1)
        self_output = paddle.concat([word_self_output, entity_self_output], axis=1)
        output = self.output(self_output, hidden_states) #######
        return output[:, : word_hidden_states.shape[1], :], output[:, word_hidden_states.shape[1]:, :]


class EntityAwareLayer(nn.Layer):
    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )
        attention_output = paddle.concat([word_attention_output, entity_attention_output], axis=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) ######

        return layer_output[:, : word_hidden_states.shape[1], :], layer_output[:, word_hidden_states.shape[1]:, :]


class EntityAwareEncoder(nn.Layer):
    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.LayerList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )
        return word_hidden_states, entity_hidden_states

        



