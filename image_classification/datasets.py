from typing import Tuple

from keras.datasets import mnist
from keras.utils import to_categorical
from numpy import float32, uint8
from numpy.typing import NDArray


def get_dataset(category="train") -> Tuple[NDArray, NDArray]:
    """根据类型获取模型数据集

    Args:
        - category (str, optional): 数据集类别. Defaults to "train".

    Returns:
        - Tuple[NDArray, NDArray]: 训练集/测试集
    """
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    if category == "train":
        train_x: NDArray[uint8]
        train_x: NDArray[float32] = (
            train_x.reshape((60000, 28, 28, 1)).astype(float32) / 255
        )

        train_y = to_categorical(train_y)

        return train_x, train_y

    test_x: NDArray[uint8]
    test_x: NDArray[float32] = (
        test_x.reshape((10000, 28, 28, 1)).astype(float32) / 255
    )

    # * 3 将输入的分类标签转换为二分类的标签
    test_y = to_categorical(test_y)

    return test_x, test_y
