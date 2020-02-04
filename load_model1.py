import os
import sys
import j_generator as jg
import constants as c
if not os.path.isdir("models"):
    os.mkdir("models")

jg.download_file_from_google_drive(c.W_ID, c.W_PATH)

model = jg.get_empty_model()
jg.load_weights_to_model(model, path=c.W_PATH)
model.save(c.NO_W_PATH)
