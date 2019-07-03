# coding=utf-8
from keras.layers import Input, Embedding, CuDNNLSTM, Dropout, Dense, Subtract, Multiply, BatchNormalization, \
    Concatenate
from keras.models import Model


class SiameseLSTM(object):
    def __init__(self, model_config):
        self.configer = model_config

    def get_model(self):
        question1 = Input(shape=(self.configer.maxlen,))
        question2 = Input(shape=(self.configer.maxlen,))

        # todo, 一个还是两个？
        embedding_op = Embedding(self.configer.max_features, self.configer.embedding_size,
                                 input_length=self.configer.maxlen)

        q1_embed = embedding_op(question1)
        q2_embed = embedding_op(question2)

        shared_lstm_1 = CuDNNLSTM(self.configer.lstm_dim, return_sequences=True)
        shared_lstm_2 = CuDNNLSTM(self.configer.lstm_dim)

        q1 = shared_lstm_1(q1_embed)
        q1 = Dropout(self.configer.dropout_rate)(q1)
        q1 = BatchNormalization()(q1)
        q1 = shared_lstm_2(q1)
        # q1 = Dropout(0.5)(q1)

        q2 = shared_lstm_1(q2_embed)
        q2 = Dropout(self.configer.dropout_rate)(q2)
        q2 = BatchNormalization()(q2)
        q2 = shared_lstm_2(q2)
        # q2 = Dropout(0.5)(q2)   # of shape (batch_size, 128)

        # 求distance (batch_size,1)
        d = Subtract()([q1, q2])
        # distance = Dot(axes=1, normalize=False)([d, d])
        # distance = Lambda(lambda x: K.abs(x))(d)
        distance = Multiply()([d, d])
        # 求angle (batch_size,1)
        # angle = Dot(axes=1, normalize=False)([q1, q2])
        angle = Multiply()([q1, q2])
        # merged = concatenate([distance,angle])

        # # magic featurues
        # magic_input = Input(shape=(train_features.shape[1],))
        # magic_dense = BatchNormalization()(magic_input)
        # magic_dense = Dense(64, activation='relu')(magic_dense)
        # # magic_dense = Dropout(0.3)(magic_dense)

        merged = Concatenate()([distance, angle])
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(256, activation='relu')(merged)  # 64
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(64, activation='relu')(merged)  # 64
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)

        is_duplicate = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[question1, question2], outputs=is_duplicate)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
