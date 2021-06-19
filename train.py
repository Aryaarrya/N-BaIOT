import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import load
from model import AnomalyDetector

import mlflow
import mlflow.keras

def train_all():

    list_attacks = ['.mirai.scan.', '.mirai.ack.', '.mirai.syn.' , '.mirai.udp.' , '.mirai.udpplain.',
                    '.gafgyt.combo.' , '.gafgyt.junk.', '.gafgyt.scan.' , '.gafgyt.tcp.' , '.gafgyt.udp.']

    results = list() #final_length will be 11 except for 2 cases

    epochs = [800,250,350,100,300,450,150,230,500]
    # epochs = [10,15,10,15,10,15,10,15,10]
    learning_rate = [0.012,0.028,0.003,0.016,0.026,0.008,0.013,0.017,0.006]
    threshold = [0.042,0.011,0.011,0.030,0.035,0.038,0.0074,0.056,0.004]
    window_size = [82,20,22,65,32,43,32,23,25]
    models_auto2 = []

    with mlflow.start_run():

        for i in range(1,10):
            print(i,"started")
            device_result = list()
            X_train = load.load_device_data_benign(i)
            X_train, X_test  = train_test_split(X_train, test_size=0.33333, random_state=1)
            X_train, X_val = train_test_split(X_train,test_size=0.5,random_state=1)
            print(X_train.shape,X_val.shape,X_test.shape)

            X_test = [X_test]

            for type_attack in list_attacks:
                try:
                    X_test.append(load.load_device_data_mal(i,type_attack))
                except Exception as e: 
                    print(e) 

            autoencoder1 = AnomalyDetector()
            autoencoder1.compile(optimizer=Adam(learning_rate=learning_rate[i-1]), loss='mse')

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            history =  autoencoder1.fit(
                x=X_train,
                y=X_train,
                epochs=epochs[i-1],
                validation_data=(X_val,X_val),
                verbose=1,
                shuffle=True
            )

            loss_train = history.history['loss']
            loss_val = history.history['val_loss']
            epochs_curr = range(1,epochs[i-1]+1)
            fig = plt.figure()
            plt.plot(epochs_curr, loss_train, 'g', label='Training loss')
            plt.plot(epochs_curr, loss_val, 'b', label='validation loss')
            plt.title('Training and Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            filename_fig = 'plots/model' + str(i) + '.png'
            plt.savefig(filename_fig)
            plt.clf()


            print("\nIOT DEVICE NUMBER ",i,"\n")
            training_losses = tf.keras.losses.mse(X_train, autoencoder1(X_train))
            threshold_curr = np.mean(training_losses)+np.std(training_losses)
            print("threshold calculated is = ",threshold_curr)

            def predict(x, threshold=threshold_curr, window_size=window_size[i-1]):
                print("threshold = ",threshold," window size: ",window_size, end = " ")
                x = scaler.transform(x)
                loss = tf.keras.losses.mse(x,autoencoder1(x))
                predictions = tf.math.greater(loss, threshold)
                # Majority voting over `window_size` predictions
                return np.array([np.mean(predictions[ii-window_size:ii]) > 0.5
                                for ii in range(window_size, len(predictions)+1)])

            def print_stats(data, outcome):
                print(f"Shape of data: {data.shape}", end=" ")
                perf = np.mean(outcome)*100
                print(f"Detected anomalies: {perf}%",end=" ")
                print("\n")
                return perf


            for jj, x in enumerate(X_test):
                print("DATASET NUMBER ",jj+1)
                outcome = predict(x,threshold[i-1],window_size[i-1])
                perf1 = print_stats(x, outcome)
                outcome2 = predict(x,window_size=window_size[i-1])
                perf2 = print_stats(x,outcome2)
                if not jj==0:
                    mlflow.log_metric("malicious"+str(i)+'_'+str(jj),perf2)
                device_result.append(perf2)


            results.append(device_result)


            mlflow.log_metric("benign"+str(i), device_result[0])
            mlflow.log_metric("train loss"+str(i),history.history['loss'][-1])
            mlflow.log_metric("val loss"+str(i),history.history['val_loss'][-1])

            mlflow.keras.log_model(autoencoder1, "model"+str(i))
            
            del autoencoder1
    return results


    

