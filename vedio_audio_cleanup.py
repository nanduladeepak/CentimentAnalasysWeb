import os
import librosa
import cv2
import pandas as pd
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
threshold = 400

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(data, sample_rate):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    # data, sample_rate = librosa.load(path)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    
    
    return result

def detect_and_crop_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cropped_faces = [gray[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return cropped_faces

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def clean_video_and_audio(filePath):
    print('File Recieved')
    # Load audio
    audio, sr = librosa.load(filePath, sr=None, res_type='kaiser_best')
    print('audio loaded')
    # Open the video file
    cap = cv2.VideoCapture(filePath)
    print('vedio loaded')
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames per 3 seconds
    frames_per_3_seconds = int(fps * 3)

    # Variables to accumulate frames and audio
    video_frames_accumulator = []
    raw_video_frames_accumulator = []
    audio_accumulator = []
    raw_audio_accumulator = []
    print('data pre processing')
    # Iterate through the video and accumulate frames and audio
    for i in range(0, total_frames, frames_per_3_seconds):
        print(f'\r step {i}/{total_frames}  ', end='', flush=True)
        start_frame = i
        end_frame = min(i + frames_per_3_seconds, total_frames)

        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read frames
        frames = []
        full_frames = []
        prev_frame = np.zeros((48, 48))
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                full_frames.append(frame)
                cropped_frame = detect_and_crop_face(frame, face_cascade)
                
                if len(cropped_frame) != 0:
                    current_frame = cv2.resize(
                        cropped_frame[0], (48, 48), interpolation=cv2.INTER_CUBIC)
                    error = mse(prev_frame, current_frame)
                    if error > threshold:
                        frames.append(current_frame)
                        prev_frame = current_frame
        if len(frames) == 0:
                frames.append(np.zeros((48, 48)))
        # Extract the corresponding audio segment
        audio_segment = audio[int(start_frame / fps * sr):int(end_frame / fps * sr)]

        audio_features = get_features(audio, sr)

        # Accumulate frames and audio
        video_frames_accumulator.extend(frames)
        raw_video_frames_accumulator.extend(full_frames)
        # audio_accumulator.extend(audio_features)
        # raw_audio_accumulator.extend(audio_segment)

    # Convert accumulated frames and audio to numpy arrays
    video_frames_array = np.array(video_frames_accumulator)
    audio_array = np.array(audio_features)
    raw_video_frames_array = np.array(raw_video_frames_accumulator)
    raw_audio_array = np.array(audio_segment)
    print('data storing to dataframe')
    # Create a DataFrame
    df = pd.DataFrame({
            'raw_vedio': [raw_video_frames_array],
            'raw_audio': [raw_audio_array],
            'video_data': [video_frames_array],
            'audio_data': [audio_array]
        })

    # Release the video capture
    cap.release()

    return df
