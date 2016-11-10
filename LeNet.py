from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import h5py
import os, pickle

def NeLet5(let_img_width = 32, let_img_height = 32, let_channel = 3, class_num = 10):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(let_channel, let_img_width, let_img_height),
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))

    sgd = SGD(lr = 1e-2, decay = 1e-6, momentum = 0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    
    # corel data
    train_dir = 'preTrainData/total_sampled_train'
    test_dir = 'preTrainData/total_sampled_test'
    class_num = 13
    vgg_img_width, vgg_img_height, vgg_channel = (224, 224, 3)
    let_img_width, let_img_height, let_channel = (14, 14, 512)
    train_batch_size = 100
    test_batch_size = 100

    NeLet_model = NeLet5(let_img_width, let_img_height, let_channel, class_num = class_num)
    # train
    X = np.empty((train_batch_size, let_channel, let_img_width, let_img_height), dtype = 'float32')
    y = np.empty((train_batch_size, class_num), dtype = 'uint8')

    test_X = np.empty((test_batch_size, let_channel, let_img_width, let_img_height), dtype = 'float32')
    test_y = np.empty((test_batch_size, ), dtype = 'uint8')

    epoch = 100; batch_count = 0;
    for e in range(epoch):
        have_load = 0; batch_count = 0
        for pic in os.listdir(train_dir):
            index = have_load % train_batch_size
            with open(train_dir + '/' + pic, 'rb') as f:
                X[index] = pickle.load(f)
                X[index] = X[index] / np.max(X[index])
            
            c = int(pic.split('_')[0])
            label = np.array([0] * class_num)
            label[c] = 1
            label.shape = (class_num, 1)
            label = np.transpose(label)
            y[index] = label
            
            if index == train_batch_size - 1:
                batch_count += 1
                print 'train', 'epoch', e, 'batch', batch_count
                NeLet_model.fit(X, y, nb_epoch = 1, shuffle = True, verbose = 2, 
                                batch_size = np.shape(X)[0])
            have_load += 1
        
        NeLet_model.save_weights('vgg_pretrain_weight_total_sampled.h5')

        # test
        all_count = 0; right_count = 0; have_load = 0; batch_count = 0
        for pic in os.listdir(test_dir):
            index = have_load % test_batch_size
            with open(test_dir + '/' + pic, 'rb') as f:
                test_X[index] = pickle.load(f)
                test_X[index] = test_X[index] / np.max(test_X[index])
            
            c = int(pic.split('_')[0])
            test_y[index] = c
            
            if index == test_batch_size - 1:
                batch_count += 1
                predict_y = np.argmax(NeLet_model.predict(test_X), axis = 1)
                right_count += sum(test_y == predict_y)
                all_count += test_batch_size
                print 'test', 'epoch', e, 'batch', batch_count
                print 'right', right_count, 'all', all_count, float(right_count)/all_count

            have_load += 1; 