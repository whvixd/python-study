from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

model = Sequential()
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 可以简单地使用 .add() 来堆叠模型：
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))


# 在完成了模型的构建后, 可以使用 .compile() 来配置学习过程：
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train,epochs=5, batch_size=32)

# 只需一行代码就能评估模型性能：
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


# 或者对新的数据生成预测：
classes = model.predict(x_test, batch_size=128)
