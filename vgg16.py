from __future__ import print_function
import keras  # 基于keras的训练模式，解决序列模型的一些问题
from keras.datasets import cifar100  # 训练集来自cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers, regularizers
import numpy as np
import torch
# from keras.layers.core import Lambda
# from keras import backend as K

from PIL import Image
import os
import re
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import sys

eps = 1e-7

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class cifar100vgg:
    def __init__(self, train = True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar100vgg.h5')


    def build_model(self):
        # 神经网络构建，采用十折交叉验证

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same', input_shape = self.x_shape, kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))  # 防止过学习

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(128, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(256, (3, 3), padding = 'same',kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size = (2, 2)))


        model.add(Conv2D(512, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size = (2, 2)))


        model.add(Conv2D(512, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer = regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self, X_train, X_test):
        # 标准化处理
        mean = np.mean(X_train, axis = (0,1,2,3))
        std = np.std(X_train, axis = (0, 1, 2, 3))
        print(mean, std)
        X_train = (X_train - mean) / (std + eps)
        X_test = (X_test - mean) / (std + eps)
        return X_train, X_test

    def normalize_production(self, x):
        # 过程中标准化处理
        mean = 121.936
        std = 68.389
        return (x - mean) / (std + eps)

    def predict(self, x, normalize=True, batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x, batch_size)

    def train(self, model):
        # 训练感知器
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1  # 学习率
        lr_decay = 1e-6
        lr_drop = 20

        # 数据
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # 数据预处理
        datagen = ImageDataGenerator(
            featurewise_center = False,  # 中位数置0
            samplewise_center = False,  # 中位数置为0
            featurewise_std_normalization = False,  # 分割数据集
            samplewise_std_normalization = False,  # 分割数据集
            zca_whitening = False,  # ZCA白化
            rotation_range=15,  # 随机旋转图片
            width_shift_range=0.1,  # 随机水平平移
            height_shift_range=0.1,  # 随机竖直平移
            horizontal_flip=True,  # 随机镜像翻转
            vertical_flip=False)  # 不采用上下翻转（怕图片可识别度不高）
        datagen.fit(x_train)

        # 优化
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        historytemp = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch = x_train.shape[0] // batch_size,
            epochs = maxepoches,
            validation_data = (x_test, y_test),
            callbacks = [reduce_lr],
            verbose = 2)
        model.save_weights('cifar100vgg.h5')
        return model

def resize_image(image_path, w, h):
    image_name = re.sub(r'.*[/\\]', '', image_path)
    outfile = re.sub(image_name, '', image_path)+'32_32.jpg'
    #print(output_name)
    #input()
    #outfile = "C:/Users/28278/Desktop/a.jpg"
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    img = Image.open(image_path)
    img.resize((w, h), Image.ANTIALIAS).save(outfile, quality=95)
    return outfile

def unpickle():
    import pickle
    with open("meta", 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def get_labels():
    return unpickle()['fine_label_names']

def pic_handler(path, show_pictures=False):
    path = path.replace('\\','/')
    outfile = resize_image(path, w = 32, h = 32)
    lena = mpimg.imread(outfile)  # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    if lena.shape != (32, 32, 3):
        print("图片尺寸有问题，请更换图片尝试，若多次尝试失败，请联系作者")
        sys.exit(0)
    if show_pictures:
        plt.imshow(lena) # 显示图片
        plt.axis('off') # 不显示坐标轴
        plt.show()
    os.remove(outfile)
    return lena

def pic_to_label(path, show_pictures=False):
    labels = unpickle()
    my_model = cifar100vgg(train = False)
    pic = pic_handler(path, show_pictures).astype('float32')
    pic = np.expand_dims(pic, axis = 0)
    predicted_x = my_model.predict(x = pic)
    return(labels['fine_label_names'][np.argmax(predicted_x, 1)[0]])


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

model = cifar100vgg()

predicted_x = model.predict(x_test)
residuals = (np.argmax(predicted_x, 1) != np.argmax(y_test, 1))
loss = sum(residuals) / len(residuals)
print("the validation 0/1 loss is: ", loss)