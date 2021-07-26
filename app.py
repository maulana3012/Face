import json,time
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
from flask_mysqldb import MySQL
from flask_socketio import SocketIO
from core_service.facerecognition import Recognizer
import MySQLdb.cursors
import requests
import bcrypt
import base64,cv2
import os,urllib.request
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import hashlib
import numpy as np
import datetime

UPLOAD_FOLDER = 'static/_asset/_upload'
DIR_FILE = '_asset/_upload/'

app = Flask(__name__)

app.secret_key = 'La12127654~!'
app.config['DIR_FILE'] = DIR_FILE
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_tesis'

mysql = MySQL(app)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("face_detector\mask_detector.model")
socketio = SocketIO(app)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def get_str_date():
    return datetime.datetime.now().strftime("%d/%m/%Y")

def get_str_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/systesis/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM tb_access_karyawan WHERE EMAIL = %s', (username,))
        account = cursor.fetchone()
        if len(account)>0:
            if account['STATUS']==1:
                if bcrypt.hashpw(password, account["TEMPORARY_PASSWORD"].encode('utf-8')) == account["TEMPORARY_PASSWORD"].encode('utf-8'):
                    session['loggedin'] = True
                    session['id'] = account['NIK']
                    session['username'] = account['FULLNAME']
                    session['password'] = account['TEMPORARY_PASSWORD']
                    session['nik'] = account['NIK']
                    return redirect(url_for('home'))
                else:
                    msg = 'Incorrect username/password!'
                    return render_template('index.html', msg=msg)
            else:
                msg = 'Username not Active'
                return render_template('index.html', msg=msg)
        else:
            msg = 'Error user not found'
    else:
        return render_template('index.html', msg=msg)

@app.route('/systesis/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))

@app.route('/systesis/home')
def home():
    msg = []
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM tb_access_karyawan')
        account = cursor.fetchone()
        if account['TEMPORARY_PASSWORD'] == session['password']:
            msg = 'Please change password, password temporary 1 days'
            return render_template('home.html', username=session['username'], msg=msg)
        else:
            return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/systesis/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT access.NIK, access.TEMPORARY_PASSWORD, access.FULLNAME,access.GAMBAR, karyawan.EMAIL, karyawan.NOMOR_HANDPHONE, karyawan.JENIS_KELAMIN FROM tb_karyawan as karyawan INNER JOIN tb_access_karyawan as access ON access.NIK=karyawan.NIK WHERE access.NIK = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/systesis/update_profil' , methods=['GET', 'POST'])
def update_profil():
    if 'loggedin' in session:
        msg = ''
        if request.method == 'POST' and 'password' in request.form and 'file' in request.files:
            if 'file' not in request.files:
                msg = 'No file part'
                return render_template('profile.html', msg=msg)
            password = request.form['password']
            salt = os.urandom(32)
            passhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            file = request.files['file']
            if file.filename == '':
                msg = 'No image selected for uploading'
                return render_template('profile.html', msg=msg)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                dirfile = os.path.join(app.config['DIR_FILE'], filename)
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cursor.execute('UPDATE tb_access_karyawan SET PASSWORD = %s, GAMBAR = %s WHERE ID = %s',(passhash, dirfile, session['id'],))
                mysql.connection.commit()
                msg = 'Image successfully uploaded and displayed below'
                return redirect(url_for('profile'))
            else:
                msg = 'Allowed image types are -> png, jpg, jpeg, gif'
                return render_template('profile.html', msg=msg)
    return redirect(url_for('login'))



@app.route('/systesis/eform')
def eform():
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM tb_access_karyawan WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('eform.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/systesis/test')
def test():
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM tb_access_karyawan WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('test.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/systesis/in_test' , methods=['GET', 'POST'])
def in_test():
    msg = ''
    if request.method == 'POST' and 'nik' in request.form and 'izin' in request.form and 'des' in request.form and 'time' in request.form:
        nik = request.form['nik']
        izin = request.form['izin']
        des = request.form['des']
        time = request.form['time']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO tb_jam_absen VALUES (NULL,%s,%s,%s,%s,%s)', (nik, izin, des, time,0))
        mysql.connection.commit()
        msg = 'You have successfully registered!'
        return render_template('test.html', msg=msg)


@app.route('/systesis/cam/')
def cam():
    camera = request.args.get("camera")
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT access.NIK, access.FULLNAME,karyawan.DEVISI, karyawan.JABATAN, karyawan.TANGGAL_BAERGABUNG FROM tb_karyawan as karyawan INNER JOIN tb_access_karyawan as access ON access.NIK=karyawan.NIK  WHERE access.NIK = %s', (session['id'],))
        account = cursor.fetchone()

        return render_template('camera.html', account=account)
    return redirect(url_for('login'))

@app.route('/systesis/absen',methods =["GET", "POST"])
def absen():
    if 'loggedin' in session:
        if request.method == "POST":
            type = request.form.get('izin')
            desk = request.form.get('Deskripsi')
            tanggal = request.form.get('tgl_pengajuan')
            masuk = request.form.get('masuk')
            pulang = request.form.get('keluar')
            akurasi = request.form.get('akurasi')
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('INSERT INTO tb_jam_absen VALUES (NULL,%s,%s,%s,%s,%s,%s,%s,%s)',(session['id'], type, desk, tanggal, akurasi, masuk, akurasi, 1))
            mysql.connection.commit()
        return render_template('eform.html')

def gen():
    cap = VideoStream(src=0).start()
    while True:
        frame = cap.read()
        frame = imutils.resize(frame, width=400)
        ret, buffer = cv2.imencode('.jpg', frame)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            snap = buffer.tobytes()
            if (label == 'Mask'):
                socketio.emit("prediction", {
                    # 'frame': snap,
                    'date': get_str_date(),
                    'time': get_str_time(),
                    'akurasi': "{:.2f}%".format(max(mask, withoutMask) * 100)
                })
                socketio.sleep(0.1)

                # print(get_str_datetime())

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('server_event')
def handle_message(message):
    print('received message: ', message)
    time.sleep(1)
    socketio.emit("client_event", "Hello from server")

if __name__ == '__main__':
    app.secret_key = "La12127654~!"
    app.run(debug=True)