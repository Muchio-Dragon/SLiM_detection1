import tensorflow as tf
from typing import List
from .common_layer import FeedForwardNetwork, ResidualNormalizationWrapper, LayerNormalization
from .embedding import TokenEmbedding, AddPositionalEncoding
from .attention import SelfAttention

PAD_ID = 0


class BinaryClassificationTransformer(tf.keras.models.Model):
    '''
    2値分類用のTransformer モデルです。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int = 4,
            head_num: int = 8,
            hidden_dim: int = 512,
            dropout_rate: float = 0.1,
            output_bias = 'zeros',
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.output_bias=output_bias

        self.encoder = Encoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.dense_layer = tf.keras.layers.Dense(
                            hidden_dim, activation='tanh')
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid',
                            bias_initializer=self.output_bias)

    def build_graph(self, name='transformer') -> None:
        '''
        学習/推論のためのグラフを構築します。
        '''
        with tf.name_scope(name):
            self.is_training = tf.compat.v1.placeholder(
                        dtype=tf.bool, name='is_training')
            # [batch_size, max_length]
            self.encoder_input = tf.compat.v1.placeholder(
                        dtype=tf.int32, shape=[None, None],
                        name='encoder_input')

            self.call(
                encoder_input=self.encoder_input,
                training=self.is_training,
            )

    def call(self, encoder_input: tf.Tensor,
             training: bool) -> tf.Tensor:
        enc_attention_mask = \
                self._create_enc_attention_mask(encoder_input)

        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=enc_attention_mask,
            training=training,
        )
       # <CLS>の部分だけを取り出す
        encoder_output = self.dense_layer(encoder_output[:, 0, :])
        encoder_output = self.dropout_layer(encoder_output,
                                            training=training)
        final_output = self.final_layer(encoder_output)
        return final_output

    def _create_enc_attention_mask(self, encoder_input: tf.Tensor):
        with tf.name_scope('enc_attention_mask'):
            batch_size, length = tf.unstack(tf.shape(encoder_input))
            # [batch_size, m_length]
            pad_array = tf.equal(encoder_input, PAD_ID)
            # shape broadcasting で [batch_size, head_num,
            #                        (m|q)_length, m_length] になる
            return tf.reshape(pad_array, [batch_size, 1, 1, length])


class Encoder(tf.keras.layers.Layer):
    '''
    トークン列をベクトル列にエンコードする Encoder です。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(vocab_size, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = SelfAttention(hidden_dim, head_num,
                    dropout_rate, name='self_attention')
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate, name='ffn')
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer,
                    dropout_rate, name='self_attention_wrapper'),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate,
                                             name='ffn_wrapper'),
            ])
        self.output_normalization = LayerNormalization()

    def call(
            self,
            input: tf.Tensor,
            self_attention_mask: tf.Tensor,
            training: bool,
    ) -> tf.Tensor:
        '''
        モデルを実行します

        :param input: shape = [batch_size, length]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input,
                                         training=training)

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = attention_layer(query, training=training,
                                attention_mask=self_attention_mask)
                query = ffn_layer(query, training=training)
        # [batch_size, length, hidden_dim]
        return self.output_normalization(query)

