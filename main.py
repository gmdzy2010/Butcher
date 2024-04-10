from os.path import exists, join

from dotenv import dotenv_values
from keras.models import load_model

from apps.image_classification import datasets as ic_dataset
from apps.image_classification.models import train_model as ic_train
from apps.text_classification import datasets as tc_dataset
from apps.text_classification.models import train_model as tc_train
from utils.exporter import evaluate_model, save_model

config = dotenv_values(".env")
MODEL_PATH = config.get("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH is empty")


def get_model_by_name(name: str):
    model_file_path = join(MODEL_PATH, name)
    if not exists(model_file_path):
        return None, model_file_path

    model = load_model(model_file_path)

    return model, model_file_path


if __name__ == "__main__":
    # * 图像分类，使用基于 CNN 构建的模型
    model_name = "ic.keras"
    model, path = get_model_by_name(model_name)
    if not model:
        dataset = ic_dataset.get_dataset(category="train")
        model, _ = ic_train(dataset, with_summary=False)
        save_model(model, path)

    dataset = ic_dataset.get_dataset(category="test")
    acc, loss = evaluate_model(model, dataset)
    print(f"CNN accuracy: {acc}, loss: {loss}")

    # * 文本分类，使用基于 CNN 构建的模型
    model_name = "tc.keras"
    model, file_path = get_model_by_name(model_name)
    if not model:
        dataset = tc_dataset.get_dataset(category="train")
        model, _ = tc_train(dataset, with_summary=False)
        save_model(model, file_path)

    dataset = tc_dataset.get_dataset(category="test")
    acc, loss = evaluate_model(model, dataset)
    print(f"CNN accuracy: {acc}, loss: {loss}")
