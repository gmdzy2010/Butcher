from keras import layers, losses, models, optimizers


def train_model(dataset, with_summary=True):
    """编译并训练 CNN 模型

    该模型由
    - 3个卷积层/最大池化层对作为卷积基，提取特征
    - 2个全连接层作为顶层分类器，对结果进行分类

    Args:
        dataset (_type_): 训练数据集
        with_summary (bool, optional): 是否在标准输出打印模型架构信息，默认输出.

    Returns:
        _type_: 模型和模型训练的历史数据
    """
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
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

    if with_summary:
        model.summary()

    model.compile(
        optimizer=optimizers.RMSprop(),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    train_x, train_y = dataset
    history = model.fit(
        train_x,
        train_y,
        epochs=5,
        batch_size=64,
    )

    return model, history
