# coding=utf-8
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Dot, Lambda, Permute, TimeDistributed, Add, Multiply, \
    Concatenate, GlobalAvgPool1D, GlobalMaxPool1D, BatchNormalization
from keras.activations import softmax
from keras.optimizers import Adam


class DecomposableAttention(object):
    def __init__(self, model_config, projection_hidden=0):
        self.configer = model_config
        self.projection_hidden = projection_hidden

    def _unchanged_shape(self, input_shape):
        "Function for Lambda layer"
        return input_shape

    def _substract(self, input_1, input_2):
        "Substract element-wise"
        neg_input_2 = Lambda(lambda x: -x, output_shape=self._unchanged_shape)(input_2)
        out_ = Add()([input_1, neg_input_2])
        return out_

    def _submult(self, input_1, input_2):
        "Get multiplication and subtraction then concatenate results"
        mult = Multiply()([input_1, input_2])
        sub = self._substract(input_1, input_2)
        out_ = Concatenate()([sub, mult])
        return out_

    def _soft_attention_alignment(self, input_1, input_2):
        "Align text representation with neural soft attention"
        attention = Dot(axes=-1)([input_1, input_2])
        w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                         output_shape=self._unchanged_shape)(attention)
        w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                         output_shape=self._unchanged_shape)(attention))
        in1_aligned = Dot(axes=1)([w_att_1, input_1])
        in2_aligned = Dot(axes=1)([w_att_2, input_2])
        return in1_aligned, in2_aligned

    def _time_distributed(self, input_, layers):
        "Apply a list of layers in TimeDistributed mode"
        out_ = []
        node_ = input_
        for layer_ in layers:
            node_ = TimeDistributed(layer_)(node_)
        out_ = node_
        return out_

    def _apply_multiple(self, input_, layers):
        "Apply layers to input then concatenate result"
        if not len(layers) > 1:
            raise ValueError('Layers list should contain more than 1 layer')
        else:
            agg_ = []
            for layer in layers:
                agg_.append(layer(input_))
            out_ = Concatenate()(agg_)
        return out_

    def get_model(self):

        # Based on: https://arxiv.org/abs/1606.01933
        q1 = Input(name='q1', shape=(self.configer.maxlen,))
        q2 = Input(name='q2', shape=(self.configer.maxlen,))

        # todo: 1/2?
        embedding_op = Embedding(self.configer.max_features, self.configer.embedding_size,
                                 input_length=self.configer.maxlen)

        q1_embed = embedding_op(q1)
        q2_embed = embedding_op(q2)

        # Projection
        projection_layers = []
        if self.projection_hidden > 0:
            projection_layers.extend([
                Dense(self.projection_hidden, activation='elu'),
                Dropout(rate=self.configer.dropout_rate),
            ])
        projection_layers.extend([
            Dense(self.configer.dense_dim, activation=None),
            Dropout(rate=self.configer.dropout_rate)
        ])

        q1_encoded = self._time_distributed(q1_embed, projection_layers)
        q2_encoded = self._time_distributed(q2_embed, projection_layers)

        # Attention
        q1_aligned, q2_aligned = self._soft_attention_alignment(q1_encoded, q2_encoded)

        # Compare
        q1_combined = Concatenate()([q1_encoded, q2_aligned, self._submult(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, self._submult(q2_encoded, q1_aligned)])
        compare_layers = [
            Dense(512, activation='elu'),
            Dropout(self.configer.dropout_rate),
            Dense(512, activation='elu'),
            Dropout(self.configer.dropout_rate),
        ]
        q1_compare = self._time_distributed(q1_combined, compare_layers)
        q2_compare = self._time_distributed(q2_combined, compare_layers)

        # Aggregate
        q1_rep = self._apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = self._apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])
        dense = BatchNormalization()(merged)
        dense = Dense(self.configer.dense_dim, activation='elu')(dense)
        dense = Dropout(self.configer.dropout_rate)(dense)
        dense = BatchNormalization()(dense)
        dense = Dense(self.configer.dense_dim, activation='elu')(dense)
        dense = Dropout(self.configer.dropout_rate)(dense)
        out_ = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[q1, q2], outputs=out_)
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
