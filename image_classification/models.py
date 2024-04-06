from os.path import join

from keras import Input, layers, models
from keras.losses import CategoricalCrossentropy
from keras.models import Model, Sequential
from keras.optimizers import RMSprop

from image_classification.datasets import get_train_dataset


def get_compiled_model():
    """获取编译好的模型"""
    model = models.Sequential(
        [
            Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=RMSprop(),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def train_model(model, dataset):
    """利用数据集训练模型"""
    train_data, train_labels = dataset
    model.fit(train_data, train_labels, epochs=5, batch_size=64)

    return model


def get_trained_model():
    model, dataset = get_compiled_model(), get_train_dataset()
    model = train_model(model, dataset)

    return model


def save_trained_model_to_file(
    path: str,
    name: str,
    overwrite=True,
) -> Sequential | Model:
    """保存训练好的模型到文件，默认保存为 .keras 压缩格式

    Args:
        - path (str): 文件路径
        - name (str): 文件名
        - overwrite (bool, optional): 是否覆盖同名模型文件，默认覆盖.

    Returns:
        - Sequential | Model: 训练好的模型
    """
    model, dataset = get_compiled_model(), get_train_dataset()
    model = train_model(model, dataset)
    file_path = join(path, f"{name}.keras")

    model.save(file_path, overwrite=overwrite)

    return model
