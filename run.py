# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from models.esim.model import ESIM
from models.esim.model_config import ESIMConfig
from models.siamese_lstm.model import SiameseLSTM
from models.siamese_lstm.model_config import SiameseLSTMConfig
from models.decomposable_attention.model import DecomposableAttention
from models.decomposable_attention.model_config import DecomposableAttentionConfig
from utils.load_data import load_char_data, char_index
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from keras.utils import plot_model

MODEL_SELECTION = {'esim': (ESIMConfig(), ESIM(ESIMConfig()).get_model()),
                   'siamese_lstm': (SiameseLSTMConfig(), SiameseLSTM(SiameseLSTMConfig()).get_model()),
                   'decomposable_attention': (
                       DecomposableAttentionConfig(), DecomposableAttention(DecomposableAttentionConfig()).get_model())}

CURRENT_PATH = os.path.dirname(__file__)
INTENT_DATA_PATH = os.path.join(CURRENT_PATH, 'data/intent_0.csv')


def get_origin_intent_data(intent_data_path):
    origin_intent_data = pd.read_csv(intent_data_path, header=None)
    origin_intent_list = [tmp_intent for tmp_intent in origin_intent_data[1].values]
    return origin_intent_list, origin_intent_data[0].values


def train(model, configer, restore_weights_path, train_data, dev_data):
    model.summary()
    checkpoint = ModelCheckpoint(restore_weights_path, monitor='val_acc', verbose=0,
                                 save_best_only=True, save_weights_only=True)
    early_stop = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto', baseline=None,
                               restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                               patience=3, min_lr=0.001)
    q1_train, q2_train, y_train = train_data
    q1_dev, q2_dev, y_dev = dev_data
    model.fit(x=[q1_train, q2_train],
              y=y_train,
              epochs=configer.epochs,
              shuffle=True,
              callbacks=[checkpoint, early_stop],
              validation_data=([q1_dev, q2_dev], y_dev),
              batch_size=configer.batch_size)


def main(model_name, dataset_name='LCQMC', train_flag=False, test_flag=False, predict_flag=False, evaluate_flag=False):
    configer, model = MODEL_SELECTION[model_name]
    restore_weights_path = 'saved_models/{}/{}_{}.h5'.format(model_name, model_name, dataset_name)
    train_data_path = 'data/train_{}.csv'.format(dataset_name)
    dev_data_path = 'data/dev_{}.csv'.format(dataset_name)
    test_data_path = 'data/test_{}.csv'.format(dataset_name)
    if train_flag:
        # process training data op.
        q1_train, q2_train, y_train = load_char_data(train_data_path, data_size=None)
        q1_dev, q2_dev, y_dev = load_char_data(dev_data_path, data_size=10000)
        train_data = (q1_train, q2_train, y_train)
        dev_data = (q1_dev, q2_dev, y_dev)
        # train op.
        train(model, configer, restore_weights_path, train_data, dev_data)
        return
    model.load_weights(restore_weights_path)
    if test_flag:
        # process test data op.
        q1_test, q2_test, y_test = load_char_data(test_data_path, data_size=10000)
        print(model.evaluate(x=[q1_test, q2_test], y=y_test, batch_size=1000))
    if predict_flag:
        p_pred, h_pred = char_index(['圣诞看打瞌睡的奶粉'], ['现在没打算'])
        print(model.predict([p_pred, h_pred]))
    if evaluate_flag:
        tmp_list = [
            ['啊', '啊，没有呢', '看看呀，我也买过一年多了', '哦，没有没有，我现在不需要了啊，谢谢啊', '嗯，没有了', '还没有', '呃，没有现在没打算', '哦，现在没有关注啊', '啊，这个没有换别的',
             '你们没人写错啊', '嗯没有', '呃，没有没有，我就是做资源的', '哦，没了没了哎', '啊', '嗯，没有现在还没有', '没有', '没有', '他不买，我怎么做？', '嗯没了', '嗯，没有了',
             '没有', '啊，没有谢谢'],
            ['最好的是吧', '我那是以后', '哦，我买了别的品牌，谢谢啊', '啊你们', '哦要买', '啊，好多进来好了啊', '嗯，好的再见', '然后呢', '我操咋买过的？', '好好好，买好这个几百年前',
             '刚买，买了才15万，工程，现在才打电话', '哦，那老早就买了哈，那就这样', '忘记', '你搞错了吧，再见', '嗯那好', '等会', '啊，你满意吗？', '哦，我问问回来了，现在不行了',
             '没说停停车位', '呃毛', '为什么限量吗？', '哎，你好哎，你好', '啊，蛮好的', '啊，我早就不关注了，是吧是吧，那好吧，好', '好好好，我去年就买了', '啊，我刚上来了', '嗯，确定拿好了',
             '什么', '哦，就昨天关注了，是不是？', '什么', '我支付麻烦', '拿好了，我买了很久了呀', '马上马上啊', '好了，谢谢啊', '什么', '喂', '我现在怎么搞的几年了？',
             '蛮好的，早就卖号楼', '昨天忙好了。', '呃，现在没关注了，好吧'],
            ['那，那好了啊', '啊，已经买了菜，不用那个了，谢谢啊，不关注', '拿好了', '几年前是签完了，买好了', '嗯，买好了', '我就买了。', '嗯，好的好。', '买好了', '哦，我早就买好了，好',
             '早就买好了。', '呃，我现在没空', '哎好嘞', '劳动买好了呀', '哦，已经买过了', '弄好了', '你，你现在给弄好了啊，好好再见', '买好了，已经啊', '购买奥迪奥迪是多少？',
             '哦，买了买了', '那好了，谢谢', '这早就买了，买了买，那早就来了。', '好嘞，好了', '哦买了', '开都开不了', '买了', '那你什么时候，我已经说已经早已买好了，谢谢啊', '买好了',
             '哎好的', '已经买好了', '嗯，那好再见啊', '嗯，买好了', '我买了买了', '好嘞好嘞', '嗯，那好嘞，谢谢', '哦，那好嘞', '哎，已经买好了', '那是几年前的4点多，也买过了，嗯嗯',
             '好嘞', '我早就买了啊', '那好的早就买好', '嗯，已经买好了', '哦，我买好了，拿好了，我们这边买的', '嗯，那好嘞', '嗯，那好了', '那好嘞，再见啊', '啊啊，对车早就买好了',
             '哎，已经买好了车子', '哦，我已经买其他车了', '好的', '哦，我买好了', '买了'],
            ['啊，不用了，不用谢谢', '嗯，好的，不需要了啊，再见嗯', '啊，车子都买的不知道给我已经买好啊，暂时不需要', '嗯，现在不考虑了啊，好吧，好，你现在不买了啊', '啊啊不需要', '啊，不需要',
             '哦，这个这个还没有，但是我现在不需要啊', '啊，不需要谢谢啊', '啊，嗯，不需要了，不需要了', '啊，不不不靠不考虑不考虑']]
        origin_intent_list, origin_classes_list = get_origin_intent_data(INTENT_DATA_PATH)
        index2class = {0: '还没买车', 1: '未匹配', 2: '买好车了', 3: '不需要买车'}
        origin = []
        most_sim = []
        sim_scores = []
        origin_classes = []
        sim_classes = []

        for tmp_index, tmp_list_ in enumerate(tmp_list):
            for tmp_data in tmp_list_:
                user_response = [tmp_data] * len(origin_intent_list)
                p_pred, h_pred = char_index(user_response, origin_intent_list)
                sim_scores_ = model.predict([p_pred, h_pred])
                most_sim_index = np.argmax(sim_scores_)
                if tmp_index != origin_classes_list[most_sim_index]:
                    origin.append(tmp_data)
                    most_sim.append(np.asarray(origin_intent_list)[most_sim_index])
                    sim_scores.append(sim_scores_[most_sim_index])
                    origin_classes.append(index2class[tmp_index])
                    sim_classes.append(index2class[origin_classes_list[most_sim_index]])

        result_df = pd.DataFrame({
            'origin': origin,
            'origin_class': origin_classes,
            'sim_score': sim_scores,
            'most_sim': most_sim,
            'sim_class': sim_classes
        })

        result_df.to_csv(os.path.join(CURRENT_PATH, 'eval_data/{}_{}.csv'.format(model_name, dataset_name)),
                         index=False)
        print('evaluate succeed!')


if __name__ == '__main__':
    main('decomposable_attention', 'bank', train_flag=True)
