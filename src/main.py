import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pytube import YouTube

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESS_FOLDER = 'processes'
AUDIO_FILE_NAME = 'audio.mp3'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

# Store user-specific process information
user_files = {}


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

@app.post("/process-file/")
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


@app.post("/process-url/")
async def download_audio(data: dict, user_id: str):
    print(data)
    '''Get audio URL'''
    if 'video_id' not in data or 'format' not in data:
        return JSONResponse(content={'error': 'Invalid request'}, status_code=400)

    video_id = data['video_id']
    request_format = data['format']

    if (request_format == 'youtube'):
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

# TODO: Add more audio processing

if __name__ == '__main__':
    if not os.path.exists(PROCESS_FOLDER):
        os.makedirs(PROCESS_FOLDER)
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=5000,
                log_level="info", reload=True)
