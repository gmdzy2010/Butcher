from os.path import join
from typing import Tuple

from keras.models import Model, Sequential


def save_model(model, path: str, overwrite=True) -> Sequential | Model:
    """保存训练好的模型到文件，默认保存为 .keras 压缩格式

    Args:
        - path (str): 文件路径
        - name (str): 文件名
        - overwrite (bool, optional): 是否覆盖同名模型文件，默认覆盖.

    Returns:
        - Sequential | Model: 训练好的模型
    """
    file_path = join(path)

    # ! 这里保存训练过的模型再次加载会报错，原因未知
    # * 是 Python 3.12.0 的原因
    # * https://github.com/tensorflow/tensorflow/issues/63365
    model.save(file_path, overwrite=overwrite)

    return model


def evaluate_model(model, dataset) -> Tuple[float, float]:
    x, y = dataset

    return model.evaluate(x, y)
