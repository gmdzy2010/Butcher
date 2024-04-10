from matplotlib import pyplot as plt
from numpy.typing import ArrayLike


def plot_both(x: ArrayLike, y1: ArrayLike, y2: ArrayLike, category="acc"):
    """给结果数据画图

    Args:
        - x (ArrayLike): X轴数据
        - y1 (ArrayLike): 训练精度或者损失数据
        - y2 (ArrayLike): 测试精度或者损失数据
        - category (str): 图形类别，默认是 acc(精度)
    """
    plt.plot(x, y1, "r", label=f"traning {category}")
    plt.plot(x, y2, "b", label=f"validation {category}")
    plt.title(f"Traning VS Validation {category}")
    plt.legend()
    plt.figure()


def plot_result(
    x: ArrayLike,
    t_acc: ArrayLike,
    v_acc: ArrayLike,
    t_loss: ArrayLike,
    v_loss: ArrayLike,
):
    plot_both(x, t_acc, v_acc, category="acc")
    plot_both(x, t_loss, v_loss, category="loss")
    plt.show()
