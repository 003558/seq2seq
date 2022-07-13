import configparser  # iniファイル形式のconfigを読み込む
import json          # iniファイルからリスト形式を読む場合にjson.loadsを使用する
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.utils import plot_model

#ダムタイプ　'gravity' or 'fill'
dam_type = 'gravity'

N_RNN = 10  # 1セットのデータ数
N_IN_OUT = 1 # 入力層・出力層のニューロン数
N_MID = 20  # 中間層のニューロン数

#バッチサイズ
batch = 4
#学習回数
epoch = 1000
    
def main():
    # cofigファイルを読み込む
    configfile = 'setting.ini'  # コマンドライン引数の1つ目
    cf = Set_config(f=configfile)
    
    if cf.dam_type == 'gravity':
        target_axis = 'DAMAXIS'
    elif cf.dam_type == 'fill':
        target_axis = 'STREAM'
    
    file_list_train = glob.glob(f'{cf.input_path_train}/{cf.infile}')
    file_list_test  = glob.glob(f'{cf.input_path_test}/{cf.infile}')
    
    param_list = []
    #for act in ['tanh', 'linear']:
    #    for opt in ['SGD', 'Adam', 'Adamax']:
    for act in cf.activation:
        for opt in cf.optimizer:
            param_list.append([act, opt])
    
    #########前処理#########
    N_SAMPLE = 0
    N_TEST = 0
    for target in file_list_train:
        data_all = pd.read_csv(target, header=4)
        n_sample = len(data_all)-N_RNN  # サンプル数
        N_SAMPLE += n_sample
    for target in file_list_test:
        data_all = pd.read_csv(target, header=4)
        n_test   = len(data_all)-N_RNN  # test数
        N_TEST += n_test
    
    shape_ = (N_SAMPLE, N_RNN, )
    shape_test_ = (N_TEST, N_RNN, )

    x_encoder = np.zeros(shape_)  # encoderの入力
    x_decoder = np.zeros(shape_)  # decoderの入力
    y_decoder = np.zeros(shape_)  # decoderの正解
    x_encoder_test = np.zeros(shape_test_)  # encoder(test)の入力
    
    for target in file_list_train:
        data_all       = pd.read_csv(target, header=4, names=('time', 'STREAM', 'DAMAXIS', 'U-D'))
        target_val_train = data_all[target_axis].values
        for i in range(len(data_all)-N_RNN):
            x_encoder[i] = target_val_train[i:i+N_RNN] #valueを10づつ入力
            x_decoder[i, 1:] = target_val_train[i:i+N_RNN-1]  # 最初の値は0のままでひとつ後にずらす
            y_decoder[i] = target_val_train[i:i+N_RNN]  # 正解はvalue値をそのまま入れる
        #axis_x_test    = np.linspace(data_all['TIME(s)'].min(), data_all['TIME(s)'].max(), len(data_all)) #0.01から47.0までの700要素の等差数列作成
    axis_x_test = []
    for target in file_list_test:
        data_all       = pd.read_csv(target, header=4, names=('time', 'STREAM', 'DAMAXIS', 'U-D'))
        axis_x_test.append(data_all['time'].values)
        target_val_test = data_all[target_axis].values
        n_test   = len(data_all)-N_RNN  # test数
        N_TEST += n_test
        for i in range(len(data_all)-N_RNN):
            x_encoder_test[i] = target_val_test[i:i+N_RNN] #valueを10づつ入力
        #data_cut       = data_all[data_all['TIME(s)'] <= 7.0]
        #target_val_cut = data_cut['DAMAXIS(gal)'].values
        #axis_x         = np.linspace(data_cut['TIME(s)'].min(), data_cut['TIME(s)'].max(), len(data_cut)) #0.01から7.0までの700要素の等差数列作成

    # サンプル数、時系列の数、入力層のニューロン数にreshape
    x_encoder = x_encoder.reshape(shape_+(N_IN_OUT,))
    x_decoder = x_decoder.reshape(shape_+(N_IN_OUT,))
    y_decoder = y_decoder.reshape(shape_+(N_IN_OUT,))
    x_encoder_test = x_encoder_test.reshape(shape_test_+(N_IN_OUT,))

    for param in param_list:
        activation = param[0]
        optimizers = param[1]
        
        # 訓練モデル定義と出力
        model, encoder_input, encoder_states, decoder_lstm, decoder_dense = train_model(N_RNN, N_IN_OUT, N_MID, activation, optimizers)
        #plot_model(model, show_shapes=True, show_layer_names=False)

        # 予測モデル定義
        encoder_model, decoder_model = predict_model(encoder_input, encoder_states, decoder_lstm, decoder_dense, N_IN_OUT, N_MID)
        
        #########訓練#########
        history = model.fit([x_encoder, x_decoder], y_decoder, batch_size=batch, epochs=epoch)
        loss = history.history['loss']
        plt.plot(np.arange(len(loss))[2:], loss[2:])
        plt.savefig('./#output/loss_batch-{}_epoch-{}.png'.format(activation, optimizers))
        #plt.show()
        plt.close()
        
        loss_df = pd.DataFrame(list(zip(np.arange(len(loss)), loss)), columns=['before', 'after'])
        loss_df.to_csv('./#output/loss.csv')
        
        i = 0
        for x_list in axis_x_test:
            #########予測#########
            x_val = []
            res_b = []
            res_a = []
            for i in range(0, len(x_list)-N_RNN, N_RNN):
                x_test = x_encoder_test[i:i+1] # 入力を一部取り出す(x_encoderは40.10,1の3次元配列で、1次元目がdemo_indicesの配列を10個取り出している)
                #y_test = predict(x_test)

                state_value = encoder_model.predict(x_test)  # encoderにデータを投げて状態(htとメモリセル)取得
                y_decoder = np.zeros((1, 1, 1)) # 出力の値
                predicted = [] # 変換結果
                for j in range(N_RNN):
                    y, h, c = decoder_model.predict([y_decoder] + state_value)  # 前の出力と状態を渡す
                    y = y[0][0][0]
                    predicted.append(y)
                    y_decoder[0][0][0] = y  # 次に渡す値
                    state_value = [h, c] # 次に渡す状態
                y_test = predicted

                for k in range(len(x_test)):
                    x_val.append(x_list[i:i+N_RNN][k])
                    res_b.append(x_test.reshape(-1)[k])
                    res_a.append(y_test[k])

            mahala_dist = score(res_a, res_b)

            fig, axes = plt.subplots(nrows=2, figsize=(40,10))
            axes[0].plot(x_val, res_b, color="b") # 変換前（青）
            axes[0].plot(x_val, res_a, color="g") # 変換後（赤）
            axes[1].plot(x_val, mahala_dist, color="r")
            plt.savefig(f'./#output/res_act-{activation}_opt-{optimizers}_{i}.png')
            #plt.show()
            plt.close()
            res_df = pd.DataFrame(list(zip(res_b, res_a)), columns=['before', 'after'])
            res_df.to_csv(f'./#output/res_act-{activation}_opt-{optimizers}_{i}.csv')
            i += 1
        
def Set_config(f):  
    '''
    configファイル '*.ini' の値を読み込む（デフォルトはconfig.ini）
        args
            f:configファイル名
        return
            cf:configファイルからの読み取り値を設定したオブジェクト
    '''
    cf = configparser.ConfigParser()
    try:
        cf.read(f, encoding='utf-8')
    except:
        cf.read(f, encoding='shift-jis')
    
    # [General]
    #cf.title       = cf.get('General', 'title')

    # [Path]
    cf.input_path_train  = cf.get('Path', 'input_path_train')
    cf.input_path_test   = cf.get('Path', 'input_path_test')
    cf.infile      = cf.get('Path', 'infile')
    cf.output_path = cf.get('Path', 'output_path')
    
    # [PARAM]
    cf.dam_type = cf.get('PARAM', 'dam_type')
    cf.activation = eval(cf.get('PARAM', 'activation'))
    cf.optimizer = eval(cf.get('PARAM', 'optimizer'))

    # 出力ファイルの接頭語          
    # cf.headname      = os.path.splitext(os.path.basename(f))[0]

    return cf

#########訓練モデル定義#########
def train_model(N_RNN, N_IN_OUT, N_MID, activation, optimizers):
    # input
    encoder_input = Input(shape=(N_RNN, N_IN_OUT)) # encoderの入力層
    decoder_input = Input(shape=(N_RNN, N_IN_OUT)) # decoderの入力層

    # encoder
    # return_stateをTrueにすることで、状態（htとメモリセル）が得られる。return_sequnceは不要
    encoder_output, state_h, state_c = LSTM(N_MID, return_state=True)(encoder_input)  # encoder LSTMの最終出力、状態(ht)、状態(メモリセル)
    encoder_states = [state_h, state_c] # LSTM結果のencoder_stateをdecoderのLSTM中間状態に渡す

    # decoder
    decoder_lstm = LSTM(N_MID, return_sequences=True, return_state=True)  # return_stateをTrueにすることで、状態（htとメモリセル）が得られる。
    decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)  # encoderから得る状態を使用。状態(htとメモリセル)は不要
    decoder_dense = Dense(N_IN_OUT, activation=activation) # 予測で再利用のために全結合を定義
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)  # 入力と出力を設定し、Modelクラスでモデルを作成
    model.compile(loss="mean_squared_error", optimizer=optimizers)
    model.summary()

    return model, encoder_input, encoder_states, decoder_lstm, decoder_dense

#########予測モデル定義#########
def predict_model(encoder_input, encoder_states, decoder_lstm, decoder_dense, N_IN_OUT, N_MID):
    # encoderのモデルを構築
    encoder_model = Model(encoder_input, encoder_states)

    # decoderのモデルを構築
    decoder_input = Input(shape=(1, N_IN_OUT)) # (1, 1)

    # n_midは中間層のニューロン数(今回は20)
    # 状態(ht)と状態(メモリセル)の入力定義
    decoder_state_in = [Input(shape=(N_MID,)), Input(shape=(N_MID,))]

    decoder_output, decoder_state_h, decoder_state_c = \
        decoder_lstm(decoder_input, initial_state=decoder_state_in) # 既存の学習済みLSTM層を使用
    decoder_states = [decoder_state_h, decoder_state_c]

    decoder_output = decoder_dense(decoder_output) # 既存の学習済み全結合層を使用
    decoder_model = Model([decoder_input] + decoder_state_in, [decoder_output] + decoder_states) # リストを+で結合

    return encoder_model, decoder_model

#########予測#########
#def predict(x_test):
#    state_value = encoder_model.predict(x_test)  # encoderにデータを投げて状態(htとメモリセル)取得
#    y_decoder = np.zeros((1, 1, 1)) # 出力の値
#    predicted = [] # 変換結果
#    for i in range(N_RNN):
#        y, h, c = decoder_model.predict([y_decoder] + state_value)  # 前の出力と状態を渡す
#        y = y[0][0][0]
#        predicted.append(y)
#        y_decoder[0][0][0] = y  # 次に渡す値
#        state_value = [h, c] # 次に渡す状態
#    return predicted

#異常度算定(誤差の正規分布)
def score(res_a, res_b):
    errors = np.abs(np.array(res_b) - np.array(res_a))
    mean = sum(errors)/len(errors)
    cov = 0
    for e in errors:
        cov += (e - mean)**2
    cov /= len(errors)
    mahala_dist = []
    for e in errors:
        mahala_dist.append(Mahala_distantce(e, mean, cov))
    return mahala_dist

# calculate Mahalanobis distance
def Mahala_distantce(x,mean,cov):
    return (x - mean)**2 / cov

if __name__ == '__main__':
    main()
    
    