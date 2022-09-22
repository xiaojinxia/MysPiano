# -- coding: utf-8 --
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from utils import *
from keras.models import load_model
import tensorflow as tf
import calendar, time, os
import json

app = Flask(__name__)
CORS(app, supports_credentials=True)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

path = "D:/Project/python/MysPiano5/"


def extract(filename, sigma):
    songs = read_midi_from_file(filename)
    time_step = 0.25

    encoded_song = []
    notes_in_song = None
    try:
        s2 = instrument.partitionByInstrument(songs)
        notes_in_song = s2.parts[0].recurse()
    except:
        notes_in_song = songs.flat.notes

    for element in notes_in_song:

        if isinstance(element, note.Note):
            m_event = element.pitch.midi
            time = element.duration.quarterLength
            # print(m_event)
            for step in range(0, int(time / time_step)):
                if step == 0:
                    encoded_song.append(str(m_event))
                else:
                    encoded_song.append("_")


        elif isinstance(element, note.Rest):
            m_event = "r"
            time = element.duration.quarterLength
            # print(m_event)

            for step in range(0, int(time / time_step)):
                if step == 0:
                    encoded_song.append(str(m_event))
                else:
                    encoded_song.append("_")



        elif isinstance(element, chord.Chord):
            m_event = '.'.join(str(n) for n in element.normalOrder)
            time = element.duration.quarterLength
            # print(m_event)
            for step in range(0, int(time / time_step)):
                if step == 0:
                    encoded_song.append(str(m_event))
                else:
                    encoded_song.append("_")

    encoded_song = " ".join(map(str, encoded_song))
    notes_ = encoded_song

    print(notes_)

    with open(path + "/data_preparation/created_mapping_0.25", 'r') as fp:
        mappings = json.load(fp)

    l = len(mappings)

    mid_int_right = notes_to_int(notes_)

    mid_float_right = [(x - float(l / 2)) / float(l / 2) for x in mid_int_right]

    del (mid_float_right[0])
    mid_float_right.pop()
    mid_float_right_ = np.reshape(mid_float_right, (1, 100))

    decoder = load_model(path + "/model/dec.h5", compile=False)
    noise_right = decoder(mid_float_right_).numpy()

    origin_right = get_data_ARI(noise_right, sigma)

    return origin_right


# 预测序列生成音乐
def get_midistream(prediction_output):
    step_duration = 0.25
    midi_stream = stream.Stream()
    start_symbol = None
    step_counter = 1
    for i, item in enumerate(prediction_output):
        if item != "_":
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter
                if start_symbol == "r":
                    notes_rest = note.Rest(quarterLength=quarter_length_duration)
                    m_event = notes_rest
                elif ("." in start_symbol):
                    notes_in_chord = start_symbol.split(".")
                    notes_chord = []
                    for current_note in notes_in_chord:
                        new_note = note.Note(int(current_note))
                        notes_chord.append(new_note)
                    new_chord = chord.Chord(notes_chord, quarterLength=quarter_length_duration)
                    m_event = new_chord
                else:
                    notes_note = note.Note(int(float(start_symbol)), quarterLength=quarter_length_duration)
                    m_event = notes_note
                midi_stream.append(m_event)
                step_counter = 1
            start_symbol = item
        else:
            step_counter += 1
    return midi_stream


# 将mid写入本地文件中
def write_mid(content):
    b = EncodeSecret(content)
    c = b.replace(" ", "")
    list = []
    for i in c:
        if (i == "0" or i == "1"):
            e = int(i)
            list.append(e)
    list.extend([0, 0])
    list2 = [list]
    origin_left = np.array(list2)
    delta = 0.46
    sigma = 1
    noise_left = data_map_ARI(origin_left, sigma, delta)
    encoder = load_model(path + "/model/gen.h5", compile=False)

    mid_float_left = encoder.predict(noise_left)

    with open(path + "/data_preparation/created_mapping_0.25", 'r') as fp:
        mappings = json.load(fp)

    l = len(mappings)
    pred_notes = [x * float(l / 2) + float(l / 2) for x in mid_float_left[0]]

    prediction_output = ['7.10']

    for i in pred_notes:
        prediction_output.append([k for k, v in mappings.items() if v == int(i)][0])

    prediction_output.append('72')
    prediction_output.append('72')
    midi_stream = get_midistream(prediction_output)
    midi_stream.write('midi', fp='{}.mid'.format("midi_out_file"))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    content = ""
    if request.method == "GET":
        print("*************")
        print(request.args.get("data"))
        content = ''.join(request.args.get("data"))
        print("-------------")
    elif request.method == "POST":
        print("*************")
        print(request.form["data"])
        content = ''.join(request.form["data"])
        print("-------------")
    else:
        print(request.method)
    if len(content) != 14:
        return {"msg": "Number of unsupported bits"}
    write_mid(content)
    # from midi2audio import FluidSynth
    # FluidSynth(sound_font=path+"sound.sf2").midi_to_audio('midi_out_file.mid', 'output.wav')
    # return send_from_directory('', 'output.wav')
    return send_from_directory('', 'midi_out_file.mid')


@app.route('/predict_wav', methods=['GET', 'POST'])
def predict_wav():
    content = ""
    if request.method == "GET":
        print("*************")
        print(request.args.get("data"))
        content = ''.join(request.args.get("data"))
        print("-------------")
    elif request.method == "POST":
        print("*************")
        print(request.form["data"])
        content = ''.join(request.form["data"])
        print("-------------")
    else:
        print(request.method)
    if len(content) != 14:
        return {"msg": "Number of unsupported bits"}
    write_mid(content)
    from midi2audio import FluidSynth
    FluidSynth(sound_font=path + "Yamaha-Grand-Lite-v2.0.sf2").midi_to_audio('midi_out_file.mid', 'output.wav')
    return send_from_directory('', 'output.wav')


# 上传文件
@app.route('/send/file', methods=['POST'])
def send_file():
    file = request.files.get('file')
    if file is None:
        # 表示没有发送文件
        return {
            'message': "文件上传失败"
        }
    file_name = file.filename
    suffix = os.path.splitext(file_name)[-1]  # 获取文件后缀（扩展名）
    if suffix != '.mid':
        return {
            'message': "文件后缀名错误"
        }
    basePath = os.path.dirname(__file__)  # 当前文件所在路径print(basePath)
    nowTime = calendar.timegm(time.gmtime())  # 获取当前时间戳改文件名print(nowTime)
    upload_path = os.path.join(basePath, 'upload',
                               str(nowTime))  # 改到upload目录下# 注意：没有的文件夹一定要先创建，不然会提示没有该路径print(upload_path)
    upload_path = os.path.abspath(upload_path)  # 将路径转换为绝对路径print("绝对路径：",upload_path)
    file.save(upload_path + suffix)  # 保存文件

    file_path = upload_path + suffix
    sigma = 1
    origin_right = extract(file_path, sigma)
    b = origin_right.reshape(100 * sigma).tolist()
    secret = DecodeSecret(b)
    print(secret)
    return {
        'code': 200,
        'messsge': "文件上传成功",
        'decodeMsg': secret,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
