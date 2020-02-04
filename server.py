from flask import Flask, request, render_template
import sys
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import j_generator as jg
import constants as c

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
model = tf.keras.models.load_model(
    c.NO_W_PATH,
    custom_objects=None,
    compile=True)


app = Flask(__name__)


@app.route("/")
def index(for_print=[], error=0):
    return render_template('index.html', for_print=for_print, error=error)


@app.route("/rnn", methods=['POST'])
def rnn():
    command = request.form['text1']
    command2 = request.form['text2']
    command3 = request.form['text3']
    global graph

    # return request.form['text'] + " Command executed via subprocess"
    if command.isdigit():
        if int(command) > 3:
            return index(error=2)
        if command3.isdigit():
            if int(command3)>150:
                return index(error=4)
            if len(command2)>20:
                return index(error=5)

            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                for_print = jg.get_jokes(model, init_word=command2+' ', jokes_num=int(command), joke_len=int(command3))
            # keras.backend.clear_session()
            return index(for_print=for_print)
        else:
            return index(error=3)
    else:
        return index(error=1)



if __name__ == "__main__":
    app.run(debug='True')
