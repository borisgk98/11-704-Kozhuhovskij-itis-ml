import tensorflow as tf
from tensorflow.keras.utils import plot_model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормализация данных
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

number_of_layer = 2
neurons_per_layer = 100
for i in range(number_of_layer):
    model.add(tf.keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# компилируем модель
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# обучаем и сохраняем
model.fit(x=x_train, y=y_train, epochs=5)
model.save('mnist.h5')

# Сохраняем визуализацию модели
plot_model(model, to_file='model.png', show_shapes=True)

# определяем точность обучения (Accuracy)
loss, accuracy = model.evaluate(x=x_test, y=y_test)
print('\nAccuracy =', accuracy)
