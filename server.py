import uuid
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

        # Save the file to the upload folder
        file_path = os.path.join(
            app.root_path, app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Return the file path as a response
        return jsonify({'file_path': file_path, 'file_name': unique_filename})

    return jsonify({'error': 'Invalid file type'})


@app.route('/send_file/<string:filename>')
def return_file(filename):
    try:
        return send_file(app.root_path + '/uploads/' + filename)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
