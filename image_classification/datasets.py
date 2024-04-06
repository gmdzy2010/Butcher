from typing import Tuple

from keras.datasets import mnist
from keras.utils import to_categorical
from numpy import float32, uint8
from numpy.typing import NDArray


def get_train_dataset() -> Tuple[NDArray, NDArray]:
    """获取训练集数据和分类标签

    Returns:
        Tuple[NDArray, NDArray]: 训练集和数据分类标签
    """
    (train_images, train_labels), _ = mnist.load_data()
    train_images: NDArray[uint8]
    train_images: NDArray[float32] = (
        train_images.reshape((60000, 28, 28, 1)).astype(float32) / 255
    )

    train_labels = to_categorical(train_labels)

    return train_images, train_labels


def get_test_dataset() -> Tuple[NDArray, NDArray]:
    """获取训练集数据和分类标签

    Returns:
        Tuple[NDArray, NDArray]: 测试集和数据分类标签
    """
    _, (test_images, test_labels) = mnist.load_data()

    test_images: NDArray[uint8]
    test_images: NDArray[float32] = (
        test_images.reshape((10000, 28, 28, 1)).astype(float32) / 255
    )

    # * 3 将输入的分类标签转换为二分类的标签
    test_labels = to_categorical(test_labels)

    return test_images, test_labels
