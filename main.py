from os.path import exists, join

from dotenv import dotenv_values
from keras.models import load_model

from image_classification.datasets import get_test_dataset
from image_classification.models import (
    save_trained_model_to_file,
)

config = dotenv_values(".env")
model_path, model_name = config.get("MODEL_PATH"), config.get("MODEL_NAME")
if not model_path or not model_name:
    raise FileNotFoundError

model_file = join(model_path, f"{model_name}.keras")
if not exists(model_file):
    model = save_trained_model_to_file(model_path, model_name)
else:
    model = load_model(model_file)


# ! 这里保存训练过的模型再次加载会报错，原因未知
# * 是 Python 3.12.0 的原因
# * https://github.com/tensorflow/tensorflow/issues/63365
print(model.summary())  # type: ignore

test_images, test_labels = get_test_dataset()
test_loss, test_accu = model.evaluate(test_images, test_labels)  # type: ignore
print(test_loss, test_accu)
