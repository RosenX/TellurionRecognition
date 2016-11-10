from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import h5py
import os

def neLet5_model(weights_path=None, img_width=128, img_height=128, channel = 3, class_num = 10):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(channel, img_width, img_height),
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    return model

if __name__ == "__main__":
    
    # data
    train_dir = 'data/total_train'
    test_dir = 'data/total_test'
    class_num = 13

    # train_dir = 'data/data4_train'
    # test_dir = 'data/data4_test'
    # class_num = 12

    img_width, img_height, channel = (32, 32, 3)

    # train_dir = 'data/nation_train'
    # test_dir = 'data/nation_test'
    # class_num = 8

    # train_dir = 'data/caltech_train'
    # test_dir = 'data/caltech_test'
    # class_num = 101

    # pretrain
    model = neLet5_model('data/vgg16_weights.h5', img_width, img_height, channel, class_num)

    sgd = SGD(lr = 1e-2, decay = 1e-6, momentum = 0.9, nesterov=True)
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

    # train
    # X = []; y = []
    have_load = 0
    train_batch_size = 100
    # train_batch_size = len(os.listdir(train_dir)) / 100
    print 'train batch size', train_batch_size
    X = np.empty((train_batch_size, channel, img_width, img_height), dtype = 'float32')
    y = np.empty((train_batch_size, class_num), dtype = 'uint8')
    epoch = 100
    for e in range(epoch):
        print 'epoch', e
        have_load = 0
        for f in os.listdir(train_dir):
            # if have_load == 100: break
            index = have_load % train_batch_size
            # if have_load == 4: break
            pic_path = train_dir + '/%s' % f
            # c = int(f.split('.')[0]) / 100
            c = int(f.split('_')[0])
            im = cv2.resize(cv2.imread(pic_path), (img_width, img_height)).astype(np.float32) 
            im = im / 255.0
            # im[:, :, 0] -= 103.939
            # im[:, :, 1] -= 116.779
            # im[:, :, 2] -= 123.68
            im = im.transpose((2, 0, 1))
            im = np.expand_dims(im, axis = 0)
            X[index, :, :, :] = im
            label = np.array([0] * class_num)
            label[c] = 1
            label.shape = (class_num, 1)
            label = np.transpose(label)
            y[index] = label
            # print 'label is ', c
            # print model.predict(im)       
            class_p = model.predict(im)
            class_p = [round(p, 4) for p in class_p[0]]
            # print 'lable is %d, predict is %d' % (c, np.argmax(class_p)), class_p
            if index == train_batch_size - 1:
                print 'loaded number is : ', have_load
                model.fit(X, y, batch_size = train_batch_size, 
                            nb_epoch = 1, shuffle = True, verbose = 2)
            have_load += 1
            # X.append(im)
            # y.append(c)
    
        model.save_weights('vgg17_weight_total.h5')
        # model.save_weights('corel_weights.h5')

        # test
        test_batch_size = 100
        # test_batch_size = len(os.listdir(test_dir)) / 100
        print 'test batch size', test_batch_size
        test_X = np.empty((test_batch_size, channel, img_width, img_height), dtype = 'float32')
        test_y = np.empty((test_batch_size, ), dtype = 'uint8')
        all_count = 0; right_count = 0; have_load = 0
        for f in os.listdir(test_dir):
            index = have_load % test_batch_size
            all_count += 1
            pic_path = test_dir + '/%s' % f
            # c = int(f.split('.')[0]) / 100
            c = int(f.split('_')[0])
            im = cv2.resize(cv2.imread(pic_path), (img_width, img_height)).astype(np.float32)
            im = im / 255.0
            
            im = im.transpose((2, 0, 1))
            im = np.expand_dims(im, axis = 0)
            
            test_X[index, :, :, :] = im
            test_y[index] = c
            if index == test_batch_size - 1:
                predict_y = np.argmax(model.predict(test_X), axis = 1)
                right_count += sum(test_y == predict_y)
                print 'right', right_count, 'all', all_count, float(right_count)/all_count 
            have_load += 1        
        
        # print float(right_count) / all_count