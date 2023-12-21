import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware
from spleeter.separator import Separator
from xgboost import XGBClassifier
import shutil
import os
from basic_pitch.inference import predict_and_save

from pytube import YouTube

import sys
import librosa
import numpy as np
from statistics import mode
import joblib
import warnings
warnings.filterwarnings('ignore')

from sound_to_midi.monophonic import wave_to_midi

# initial and config
app = FastAPI()
separator = Separator('spleeter:5stems')
scaler = joblib.load('./models/min_max_scaler.save')
model = joblib.load('./models/xgb_5s_model.save')

# config
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
PROCESS_FOLDER = 'processes'
AUDIO_FILE_NAME = 'audio.mp3'
MIDI_FILE_NAME = 'midi.mid'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}
origins = ["*"]
min_distance = 15
song_err = []

# Store user-specific process information
user_files = {}

def save_youtube_audio(user_id: str, video_id: str):
    try:
            # Download the YouTube video audio
            youtube_url = f'https://www.youtube.com/watch?v={video_id}'
            yt = YouTube(youtube_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            # Generate a unique filename
            unique_id = str(uuid.uuid4())

            audio_path = os.path.join(
                PROCESS_FOLDER, unique_id)

            if not os.path.exists(audio_path):
                os.makedirs(audio_path)

            # Download the file to the process folder
            audio_stream.download(output_path=audio_path,
                                  filename=AUDIO_FILE_NAME)

            # Return the file path as a response
            return save_process(user_id, unique_id, audio_stream.title)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

def save_spotify_audio(user_id: str, track_url: str):
    try:
            # TODO: Download the Spotify Audio

            # Generate a unique filename
            unique_id = str(uuid.uuid4())

            audio_path = os.path.join(
                PROCESS_FOLDER, unique_id)

            if not os.path.exists(audio_path):
                os.makedirs(audio_path)

            # Download the file to the process folder
            audio_stream.download(output_path=audio_path,
                                  filename=AUDIO_FILE_NAME)

            # Return the file path as a response
            return save_process(user_id, unique_id, audio_stream.title)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

def save_process(user_id: str, process_id: str, process_name: str):
    '''Save user process and return process ID'''
    # Update user-specific file information
    user_files[user_id] = {'process_id': process_id,
                           'process_name': process_name}

    # Return the file path as a response
    return JSONResponse(content={'process_id': process_id, 'process_name': process_name})


def allowed_file(filename):
    '''Check for allowed file type'''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/process-file")
async def upload_file(file: UploadFile, user_id: str):
    '''Receive files'''
    if not file:
        return JSONResponse(content={'error': 'No file part'}, status_code=400)

    if file.filename == '':
        return JSONResponse(content={'error': 'No selected file'}, status_code=400)

    if allowed_file(file.filename):
        # Generate a unique filename
        unique_id = str(uuid.uuid4())

        # Save the file to the process folder
        file_path = os.path.join(
            PROCESS_FOLDER, unique_id)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(os.path.join(file_path, AUDIO_FILE_NAME), "wb") as f:
            f.write(file.file.read())

        # Return the unique identifier associated with the uploaded file
        return save_process(user_id, unique_id, file.filename)

    return JSONResponse(content={'error': 'Invalid file type'}, status_code=400)


@app.post("/process-url")
async def download_audio(data: dict, user_id: str):
    print(data)
    '''Get audio URL'''
    if 'content_id' not in data or 'format' not in data:
        return JSONResponse(content={'error': 'Invalid request'}, status_code=400)

    content_id = data['content_id']
    request_format = data['format']

    if (request_format == 'youtube'):
        save_youtube_audio(user_id, content_id)

@app.get("/info/{user_id}")
async def return_track_info(user_id: str):
    '''Track Identification'''
    if user_id in user_files:
        file_info = user_files[user_id]
        file_path = file_info['process_id']

        # TODO: add track identification


        return JSONResponse(content={'text': 'YAY'})
    else:
        return JSONResponse(content={'error': 'User has no uploaded files'}, status_code=404)

@app.get("/audio-file/{user_id}")
async def return_user_file(user_id: str):
    '''Send the file requested'''
    if user_id in user_files:
        file_info = user_files[user_id]
        return FileResponse(path=os.path.join(PROCESS_FOLDER, file_info['process_id'], AUDIO_FILE_NAME),
                            filename=AUDIO_FILE_NAME, media_type='application/octet-stream')
    else:
        return JSONResponse(content={'error': 'User has no uploaded files'}, status_code=404)

@app.get("/midi-file/{user_id}")
async def convert_audio_to_midi(user_id: str):
    '''Send the file requested'''
    if user_id in user_files:
        file_info = user_files[user_id]
        file_in = os.path.join(PROCESS_FOLDER, file_info['process_id'], AUDIO_FILE_NAME)
        file_out = os.path.join(PROCESS_FOLDER, file_info['process_id'], MIDI_FILE_NAME)
        y, sr = librosa.load(file_in, sr=None)
        print("Converting...", sr)
        midi = wave_to_midi(y, srate=int(sr)) # TODO: Improve MIDI conversion
        print("Done converting!")
        with open (file_out, 'wb') as f:
            midi.writeFile(f)
        return FileResponse(path=os.path.join(PROCESS_FOLDER, file_info['process_id'], MIDI_FILE_NAME),
                            filename=MIDI_FILE_NAME, media_type='application/octet-stream')
    else:
        return JSONResponse(content={'error': 'User has no uploaded files'}, status_code=404)

# TODO: Add more audio processing
@app.post("/separate")
async def separate_audio(audio_file: UploadFile = File(...)):
    if not audio_file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Only audio files with extensions .mp3, .wav, or .ogg are supported")

    file_path = f"./uploads/{audio_file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    output_path = './output/track_split/'
    os.makedirs(output_path, exist_ok=True)
    separator.separate_to_file(file_path, output_path)
    os.remove(file_path)

    return {
        'vocals_path': f'{output_path}audio_example/vocals.wav',
        'accompaniment_path': f'{output_path}audio_example/accompaniment.wav'
    }

@app.post("/audio2midi")
async def audio_to_midi(audio_file: UploadFile = File(...)):
    if not audio_file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Only audio files with extensions .mp3, .wav, or .ogg are supported")

    file_path = f"./temp/{audio_file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    
    output_path = './output/audio2midi/'
    os.makedirs(output_path, exist_ok=True)
    predict_and_save(
        [file_path],
        output_path,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
    )

    os.remove(file_path)

    return {
        'midi_path': f'{output_path}/{os.path.splitext(os.path.basename(file_path))[0]}_basic_pitch.mid',
    }

async def genre_predict(audio_file: UploadFile = File(...)):
    if not audio_file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Only audio files with extensions .mp3, .wav, or .ogg are supported")

    file_path = f"./temp/{audio_file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)
        
    y, sr = librosa.load(file_path)
    labels = ['EDM', 'Pop', 'RnB', 'Rock', 'Trap']
    predictions = []
    
     
    for i in range(2):
        # Calculate a non-negative start_time
        start_time = max(0, np.random.uniform(0.2 * len(y), 0.8 * len(y) - 30 * sr))
        
        while i > 0 and abs(start_time - prev_start_time) < min_distance * sr:
            start_time = max(0, np.random.uniform(0.2 * len(y), 0.8 * len(y) - 30 * sr))

        start_time = int(start_time)
        
        if start_time < 0:
            print(f"error: {file_path}")
            song_err.append(file_path)
        
        # split 5s
        for j in range(6):
            part_start = start_time + j * 5 * sr
            part_end = part_start + 5 * sr
            y_split = y[part_start:part_end]
            x_features = get_feature(y_split, sr)
            x_norm = scaler.transform([x_features])
            y_pred = model.predict(x_norm)
            predictions.append(labels[y_pred[0]])
        
        prev_start_time = start_time
        
    result = mode(predictions)
    
    return {"song_type": result}


def get_feature(y, sr):

    # length
    length = y.shape[0]

    # chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # rms
    rms = librosa.feature.rms(y=y)

    # spectral_centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y)

    # spectral_bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)

    # rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)

    # zero_crossing_rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # harmony and perceptr
    harmony, perceptr = librosa.effects.hpss(y=y)

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y)

    # mfcc
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)


    # concat all features
    features = [length, chroma_stft.mean(), chroma_stft.var(), rms.mean(), rms.var(), spectral_centroid.mean(), spectral_centroid.var(), spectral_bandwidth.mean(), spectral_bandwidth.var(), spectral_rolloff.mean(), spectral_rolloff.var(), zero_crossing_rate.mean(), zero_crossing_rate.var(), harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var(), tempo,
                mfccs_mean[0], mfccs_var[0], mfccs_mean[1], mfccs_var[1], mfccs_mean[2], mfccs_var[2], mfccs_mean[3], mfccs_var[3], mfccs_mean[4], mfccs_var[4], mfccs_mean[5], mfccs_var[5], mfccs_mean[6], mfccs_var[6], mfccs_mean[7], mfccs_var[7], mfccs_mean[8], mfccs_var[8], mfccs_mean[9], mfccs_var[9], mfccs_mean[10], mfccs_var[10], 
                mfccs_mean[11], mfccs_var[11], mfccs_mean[12], mfccs_var[12], mfccs_mean[13], mfccs_var[13], mfccs_mean[14], mfccs_var[14], mfccs_mean[15], mfccs_var[15], mfccs_mean[16], mfccs_var[16], mfccs_mean[17], mfccs_var[17], mfccs_mean[18], mfccs_var[18], mfccs_mean[19], mfccs_var[19]]

    return features




if __name__ == '__main__':
    if not os.path.exists(PROCESS_FOLDER):
        os.makedirs(PROCESS_FOLDER)
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000,
                log_level="info", reload=True)
