from operator import methodcaller
import os
import sys
sys.path.insert(0, 'utils')

from distutils.log import debug
from flask import Flask, request, render_template, flash, url_for, redirect
from werkzeug.utils import secure_filename


from utils import show_results
from show_results import show_results


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'


app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024




ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_files', methods = ['POST', 'GET'])
def upload_content_image():

    if len(request.form['iterations']) == 0:
        number_of_iter = 10
    else:
        number_of_iter = int(request.form['iterations'])

    file_1 = request.files['file_1']
    file_2 = request.files['file_2']

    filename_1 = secure_filename(file_1.filename)
    filename_2 = secure_filename(file_2.filename)

    file_1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_1))
    file_2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_2))


    filenames = [filename_1, filename_2]

    content_path = 'static/uploads/' + filename_1
    style_path = 'static/uploads/' + filename_2

    out_img_name = show_results(content_path, style_path, number_of_iter)
    filenames.append(out_img_name)

    for var in filenames:
        print(var)

    return render_template('sucess.html', filenames = filenames)


@app.route('/display/<filename>', methods = ['POST', 'GET'])
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug = True)