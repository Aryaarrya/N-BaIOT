import numpy as np
import pandas as pd

def load_device_data_benign(device_num):
    filepath = '/home/arya/drive/CityUniv/datasets/N-BaIOT/'
    filename = str(device_num) + ".benign.csv"
    file = filepath + filename
    with open(file, 'r') as csvfile:
        benign_df = pd.read_csv(csvfile)
        benign_df.columns = benign_df.iloc[0]
        benign_df = benign_df[1:]
        benign_iot = benign_df.to_numpy()
    print(benign_iot.shape)
    return benign_iot

def load_device_data_mal(device_num,type_attack):   #type_attack = mirai.ack
    filepath = '/home/arya/drive/CityUniv/datasets/N-BaIOT/'
    filename = str(device_num) + type_attack + "csv"
    file = filepath + filename
    with open(file, 'r') as csvfile:
        df = pd.read_csv(csvfile)
        df.columns = df.iloc[0]
        df = df[1:]
        df_iot = df.to_numpy()
    print(df_iot.shape)
    return df_iot

    
