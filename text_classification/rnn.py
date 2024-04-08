from keras.datasets import imdb
from keras.layers import Dense, Embedding, Flatten, Input
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt

max_features = 10000
max_len = 40
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=max_features)
train_x = pad_sequences(train_x, maxlen=max_len)
test_x = pad_sequences(test_x, maxlen=max_len)

model = Sequential()
model.add(Input(shape=(max_len,)))
model.add(Embedding(max_features, 8))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history = model.fit(
    train_x,
    train_y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)

test_loss, test_acc = model.evaluate(test_x, test_y)
print(test_loss, test_acc)

t_accu, v_accu = history.history["accuracy"], history.history["val_accuracy"]
t_loss, v_loss = history.history["loss"], history.history["val_loss"]
epochs = range(1, len(t_accu) + 1)
plt.plot(epochs, t_accu, "r", label="traning acc")
plt.plot(epochs, v_accu, "b", label="validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, t_loss, "r", label="Traning loss")
plt.plot(epochs, v_loss, "b", label="validation loss")
plt.title("Training and Validation loss")
plt.legend()

plt.show()
