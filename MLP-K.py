import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
# from keras.layers import Dense
from keras.utils import np_utils
from keras.layers.core import Activation, Dropout, Dense
from keras import optimizers

FILE_PRE = 'MPModel'
batch_size = 128
classes = 10
epoch = 12
img_size = 28 * 28

print('Loading Data...')
(X_train, y_train),(X_test,y_test) = mnist.load_data()

X_train = X_train.reshape(y_train.shape[0], img_size).astype('float32') / 255
X_test = X_test.reshape(y_test.shape[0], img_size).astype('float32') / 255

#对标签进行 one-hot 编码
Y_train = np_utils.to_categorical(y_train,classes)
Y_test = np_utils.to_categorical(y_test,classes)

model = Sequential([Dense(512,input_shape=(img_size,)),
                    Activation('relu'),
                    Dropout(0.2),
                    Dense(512, input_shape=(512,)),
                    Activation('relu'),
                    Dropout(0.2),
                    Dense(10,input_shape=(512,),activation='softmax')
                    ])

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=batch_size,epochs=epoch,
          verbose=1, validation_data=(X_test,Y_test))

score = model.evaluate(X_test,Y_test,verbose=0)
print('accuracy: {}'.format(score[1]))
# 保存实验模型，在 MLP 实验中，模型命名均已 MPModel+准确率的形式，储存了所有需要保存的模型信息。
if input('Do you want to save the model ?\n y/n : ') == 'y':
    model.save(FILE_PRE +'-'+ str(score[1])[2:] + '.h5')