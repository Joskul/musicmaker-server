import uuid
import os
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from pytube import YouTube

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store user-specific file information
user_files = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + '.' + \
            file.filename.rsplit('.', 1)[1].lower()
        user_id = request.args.get('user_id')

        # Save the file to the upload folder
        file_path = os.path.join(
            app.root_path, app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Update user-specific file information
        user_files[user_id] = {'file_path': file_path,
                               'file_name': unique_filename}

        # Return the unique identifier associated with the uploaded file
        return jsonify({'file_id': unique_filename})

    return jsonify({'error': 'Invalid file type'})


@app.route('/process_video', methods=['POST'])
def download_audio():
    data = request.get_json()

    if 'videoId' not in data or 'userId' not in data:
        return jsonify({'error': 'Invalid request'})

    video_id = data['videoId']
    user_id = data['userId']

    try:
        # Download the YouTube video audio
        youtube_url = f'https://www.youtube.com/watch?v={video_id}'
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + '.' + \
            (video_id+user_id).rsplit('.', 1)[0].lower() + '.mp3'

        audio_path = os.path.join(
            app.root_path, app.config['UPLOAD_FOLDER'], unique_filename)
        audio_stream.download(
            output_path=app.config['UPLOAD_FOLDER'], filename=unique_filename)

        # Update user-specific file information
        user_files[user_id] = {'file_path': audio_path,
                               'file_name': unique_filename}

        # Return the file path as a response
        return jsonify({'file_id': unique_filename})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/send_file/<string:userId>')
def return_user_file(userId):
    if userId in user_files:
        file_info = user_files[userId]
        return send_file(file_info['file_path'], as_attachment=True, download_name=file_info['file_name'])
    else:
        return jsonify({'error': 'User has no uploaded files'})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
