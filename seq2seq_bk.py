import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, LSTM, Input
from keras.utils.vis_utils import plot_model


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
def predict(x_test):
    state_value = encoder_model.predict(x_test)  # encoderにデータを投げて状態(htとメモリセル)取得
    y_decoder = np.zeros((1, 1, 1)) # 出力の値
    predicted = [] # 変換結果
    for i in range(N_RNN):
        y, h, c = decoder_model.predict([y_decoder] + state_value)  # 前の出力と状態を渡す
        y = y[0][0][0]
        predicted.append(y)
        y_decoder[0][0][0] = y  # 次に渡す値
        state_value = [h, c] # 次に渡す状態
    return predicted

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
    
    target = './#input/konk.csv'
    param_list = []
    #for act in ['tanh', 'linear']:
    #    for opt in ['SGD', 'Adam', 'Adamax']:
    for act in ['linear']:
        for opt in ['Adam']:
            param_list.append([act, opt])
    
    #########前処理#########
    data_all       = pd.read_csv(target, header=4)
    target_val_all = data_all['DAMAXIS(gal)'].values
    axis_x_test    = np.linspace(data_all['TIME(s)'].min(), data_all['TIME(s)'].max(), len(data_all)) #0.01から47.0までの700要素の等差数列作成
    
    data_cut       = data_all[data_all['TIME(s)'] <= 7.0]
    target_val_cut = data_cut['DAMAXIS(gal)'].values
    axis_x         = np.linspace(data_cut['TIME(s)'].min(), data_cut['TIME(s)'].max(), len(data_cut)) #0.01から7.0までの700要素の等差数列作成
    
    
    N_RNN = 10  # 1セットのデータ数
    N_SAMPLE = len(axis_x)-N_RNN  # サンプル数
    N_TEST   = len(axis_x_test)-N_RNN  # test数
    N_IN_OUT = 1 # 入力層・出力層のニューロン数
    N_MID = 20  # 中間層のニューロン数
    shape_ = (N_SAMPLE, N_RNN, )
    shape_test_ = (N_TEST, N_RNN, )
    
    #バッチサイズ
    batch = 4
    #学習回数
    epoch = 200
    
    x_encoder = np.zeros(shape_)  # encoderの入力
    x_decoder = np.zeros(shape_)  # decoderの入力
    y_decoder = np.zeros(shape_)  # decoderの正解
    x_encoder_test = np.zeros(shape_test_)  # encoder(test)の入力
    
    for i in range(N_SAMPLE):
        x_encoder[i] = target_val_cut[i:i+N_RNN] #valueを10づつ入力
        x_decoder[i, 1:] = target_val_cut[i:i+N_RNN-1]  # 最初の値は0のままでひとつ後にずらす
        y_decoder[i] = target_val_cut[i:i+N_RNN]  # 正解はvalue値をそのまま入れる
    for i in range(N_TEST):
        x_encoder_test[i] = target_val_all[i:i+N_RNN] #valueを10づつ入力
        
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
        plot_model(model, show_shapes=True, show_layer_names=False)

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
        
        #########予測#########
        x_val = []
        res_b = []
        res_a = []
        for i in range(0, 4690, 10):
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
                x_val.append(axis_x_test[i:i+N_RNN][k])
                res_b.append(x_test.reshape(-1)[k])
                res_a.append(y_test[k])
                
        mahala_dist = score(res_a, res_b)
        
        fig, axes = plt.subplots(nrows=2, figsize=(40,10))
        axes[0].plot(x_val, res_b, color="b") # 変換前（青）
        axes[0].plot(x_val, res_a, color="g") # 変換後（赤）
        axes[1].plot(x_val, mahala_dist, color="r")
        plt.savefig('./#output/res_batch-{}_epoch-{}.png'.format(activation, optimizers))
        #plt.show()
        plt.close()
        res_df = pd.DataFrame(list(zip(res_b, res_a)), columns=['before', 'after'])
        res_df.to_csv('./#output/res.csv')
        

        
            