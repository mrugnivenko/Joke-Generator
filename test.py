import time
import sys
import j_generator as jg
import tensorflow as tf
import constants as c

start_time = time.time()
model = tf.keras.models.load_model(
    c.NO_W_PATH,
    custom_objects=None,
    compile=True)
jokes = jg.get_jokes(model, init_word='шляпа ', jokes_num=1, joke_len=20)
print(jokes)

end_time = time.time()
print(end_time - start_time)
