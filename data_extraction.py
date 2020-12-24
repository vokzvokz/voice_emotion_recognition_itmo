import librosa
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle


def rename_list_of_files(list_of_files):
    feeling_list = []
    for item in list_of_files:
        if item[6:-16] == '02' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_calm')
        elif item[6:-16] == '02' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_calm')
        elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_happy')
        elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_happy')
        elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_sad')
        elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_sad')
        elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_angry')
        elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_angry')
        elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_fearful')
        elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_fearful')
        elif item[:1] == 'a':
            feeling_list.append('male_angry')
        elif item[:1] == 'f':
            feeling_list.append('male_fearful')
        elif item[:1] == 'h':
            feeling_list.append('male_happy')
        # elif item[:1]=='n':
        # feeling_list.append('neutral')
        elif item[:2] == 'sa':
            feeling_list.append('male_sad')

    return feeling_list


def read_as_mfcc(list_of_files):
    df = pd.DataFrame(columns=['feature'])
    bookmark = 0
    for index, y in enumerate(list_of_files):
        # print('{}/{}'.format(index, len(list_of_files)))
        if list_of_files[index][6:-16] != '01' \
                and list_of_files[index][6:-16] != '07' \
                and list_of_files[index][6:-16] != '08' \
                and list_of_files[index][:2] != 'su' \
                and list_of_files[index][:1] != 'n' \
                and list_of_files[index][:1] != 'd':
            X, sample_rate = librosa.load('RawData/' + y, res_type='kaiser_fast', duration=2.5, sr=22050 * 2,
                                          offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=13),
                            axis=0)
            feature = mfccs
            df.loc[bookmark] = [feature]
            bookmark = bookmark + 1
    # print('Done')
    return df


def read_raw_files(path='RawData/', n_files=0):
    list_of_files = os.listdir(path)
    if n_files != 0:
        list_of_files = list_of_files[0:-1:n_files]
    feelings_list = rename_list_of_files(list_of_files)
    labels = pd.DataFrame({'label': feelings_list})
    frame = read_as_mfcc(list_of_files)
    frame = pd.DataFrame(frame['feature'].values.tolist())
    df = pd.concat([labels, frame], axis=1)
    df = shuffle(df)
    df = df.fillna(0)
    return df
