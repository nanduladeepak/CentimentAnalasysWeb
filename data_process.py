import cv2
import os
import numpy as np
import librosa
import pandas as pd
from vedio_audio_cleanup import clean_video_and_audio
import tensorflow as tf

def pad_frames(vedData):
    frame_padding_amount = 100 - len(vedData)
    if(frame_padding_amount>0):
        zero_arrays = [np.zeros((48, 48)) for _ in range(frame_padding_amount)]
        vedData = np.concatenate([vedData, np.array(zero_arrays)], axis=0)
        # vedData.append([np.zeros((48, 48)) for _ in range(frame_padding_amount)])
    elif(frame_padding_amount<0):
        print(f'frame Padding is {frame_padding_amount} vedio data lenfth is {np.shape(vedData)}')
    return vedData

def getPreProcessedData(filePath):
    df = clean_video_and_audio(filePath)
    print('data reshaping')
    df['video_data_pad'] = df['video_data'].apply(pad_frames)
    X_video = np.stack(df['video_data_pad'].to_numpy())
    X_audio = np.stack(df['audio_data'].to_numpy())
    X_video_flot = X_video.astype('float32')
    X_audio_flot = X_audio.astype('float32')
    X_audio_reshape = tf.expand_dims(X_audio_flot, axis=-1)
    return [X_audio_reshape,X_video_flot]