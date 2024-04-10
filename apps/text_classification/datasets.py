from typing import Tuple

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from numpy.typing import NDArray


def get_dataset(
    category="train",
    features=10000,
    max_len=40,
) -> Tuple[NDArray, NDArray]:
    """根据类型获取模型数据集

    Args:
        - category (str, optional): 数据集类型. Defaults to "train".
        - features (int, optional): 特征数量上限. Defaults to 10000.
        - max_len (int, optional): 每条数据读取的词上限. Defaults to 40.

    Returns:
        - Tuple[NDArray, NDArray]: 训练集/测试集
    """

    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=features)
    if category == "train":
        train_x = pad_sequences(train_x, maxlen=max_len)

        return train_x, train_y

    test_x = pad_sequences(test_x, maxlen=max_len)

    return test_x, test_y
