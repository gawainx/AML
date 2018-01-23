import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers
from keras import optimizers

FILE_PRE = 'SIModel-mean_absolute_error'
batch_size = 128
classes = 10
epoch = 10
img_size = 28 * 28

print('Loading Data...')
(X_train, y_train),(X_test,y_test) = mnist.load_data()

X_train = X_train.reshape(y_train.shape[0], img_size).astype('float32') / 255
X_test = X_test.reshape(y_test.shape[0], img_size).astype('float32') / 255

#encode labels
Y_train = np_utils.to_categorical(y_train,classes)
Y_test = np_utils.to_categorical(y_test,classes)


model = Sequential([Dense(10, input_shape=(img_size,), activation='softmax'),])
model.compile(optimizer='rmsprop', loss='mean_absolute_error', metrics=['accuracy'])

print("Training...")
model.fit(X_train, Y_train,batch_size=batch_size, epochs=epoch, verbose=1, validation_data=(X_test,Y_test))

score = model.evaluate(X_test,Y_test,verbose=0)

print('accuracy: {}'.format(score[1]))

# 保存实验模型。在单层感知机模型的保存时，通过文件名体现主要的改变参数和准确率信息。
if input('Do you want to save the model ? y/n \n') == 'y':
    model.save(FILE_PRE + str(score[1])[2:] + '.h5')