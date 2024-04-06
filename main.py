from os.path import exists, join

from dotenv import dotenv_values

# from keras.models import load_model
from image_classification.datasets import get_test_dataset
from image_classification.models import (
    get_trained_model,
    save_trained_model_to_file,
)

config = dotenv_values(".env")
model_path, model_name = config.get("MODEL_PATH"), config.get("MODEL_NAME")
if not model_path or not model_name:
    raise FileNotFoundError

model_file = join(model_path, f"{model_name}.keras")
if not exists(model_file):
    save_trained_model_to_file(model_path, model_name)

# ! 这里保存训练过的模型再次加载会报错，原因未知
# model = load_model(model_file)
model = get_trained_model()
print(model.summary())

test_images, test_labels = get_test_dataset()
test_loss, test_accu = model.evaluate(test_images, test_labels)
print(test_loss, test_accu)
