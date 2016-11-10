from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import h5py
import os

def VGG_16_pretrain(weights_path=None, img_width=128, img_height=128, channel = 3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    weights_path = 'data/vgg16_weights.h5'

    f = h5py.File(weights_path)
    for k in range(len(model.layers)):
        if isinstance(model.layers[k],Convolution2D):
            attr = model.layers[k].name
            g = f[attr]
            weights = [g[attr+'_W'], g[attr+'_b']]
            model.layers[k].set_weights(weights)
    f.close()

    print('Model loaded.')

    return model

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
    
    # data
    train_dir = 'data/corel_train'
    test_dir = 'data/corel_test'
    class_num = 10
    vgg_img_width, vgg_img_height, vgg_channel = (224, 224, 3)
    let_img_width, let_img_height, let_channel = (14, 14, 512)

    # train_dir = 'data/nation_train'
    # test_dir = 'data/nation_test'
    # class_num = 8

    # train_dir = 'data/caltech_train'
    # test_dir = 'data/caltech_test'
    # class_num = 101

    # pretrain
    VGG_model = VGG_16_pretrain('data/vgg16_weights.h5', vgg_img_width, vgg_img_height, vgg_channel)
    NeLet_model = NeLet5(let_img_width, let_img_height, let_channel, class_num = class_num)
    sgd = SGD(lr = 1e-5, decay = 0.0, momentum = 0.9, nesterov=True)
    VGG_model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

    # train
    # X = []; y = []
    have_load = 0
    train_batch_size = 10
    # train_batch_size = len(os.listdir(train_dir)) / 100
    print 'train batch size', train_batch_size
    X = np.empty((train_batch_size, vgg_channel, vgg_img_width, vgg_img_height), dtype = 'float32')
    y = np.empty((train_batch_size, class_num), dtype = 'uint8')
    epoch = 100
    for e in range(epoch):
        print 'epoch is', e
        for f in os.listdir(train_dir):
            index = have_load % train_batch_size
            # if have_load == 4: break
            pic_path = train_dir + '/%s' % f
            c = int(f.split('.')[0]) / 100
            # c = int(f.split('_')[0])
            im = cv2.resize(cv2.imread(pic_path), (vgg_img_width, vgg_img_height)).astype(np.float32) 
            im[:, :, 0] -= 103.939
            im[:, :, 1] -= 116.779
            im[:, :, 2] -= 123.68
            im = im.transpose((2, 0, 1))
            im = np.expand_dims(im, axis = 0)
            # VGG_model.predict(im)
            # print VGG_model.layers[25].output
            # print np.shape(VGG_model.predict(im))

            X[index, :, :, :] = im
            label = np.array([0] * class_num)
            label[c] = 1
            label.shape = (class_num, 1)
            label = np.transpose(label)
            y[index] = label

            # class_p = VGG_model.predict(im)
            # class_p = [round(p, 4) for p in class_p[0]]
            # print 'lable is %d, predict is %d' % (c, np.argmax(class_p)), class_p
            if index == train_batch_size - 1:
                new_X = VGG_model.predict(X)
                print 'loaded number is : ', have_load
                NeLet_model.fit(new_X, y, batch_size = train_batch_size, 
                            nb_epoch = 1, shuffle = True, verbose = 2)
            have_load += 1
            # X.append(im)
            # y.append(c)
    
        # model.save_weights('caltech_weights.h5')
        VGG_model.save_weights('corel_weights.h5')
        # test
        test_batch_size = 32
        # test_batch_size = len(os.listdir(test_dir)) / 100
        print 'test batch size', test_batch_size
        test_X = np.empty((test_batch_size, channel, vgg_img_width, vgg_img_height), dtype = 'float32')
        test_y = np.empty((test_batch_size, ), dtype = 'uint8')
        all_count = 0; right_count = 0; have_load = 0
        for f in os.listdir(test_dir):
            index = have_load % test_batch_size
            all_count += 1
            pic_path = test_dir + '/%s' % f
            c = int(f.split('.')[0]) / 100
            # c = int(f.split('_')[0])
            im = cv2.resize(cv2.imread(pic_path), (vgg_img_width, vgg_img_height)).astype(np.float32)
            im[:, :, 0] -= 103.939
            im[:, :, 1] -= 116.779
            im[:, :, 2] -= 123.68
            im = im.transpose((2, 0, 1))
            im = np.expand_dims(im, axis = 0)
            
            test_X[index, :, :, :] = im
            test_y[index] = c
            if index == test_batch_size - 1:
                new_test_X = VGG_model.predict(test_X)
                predict_y = np.argmax(NeLet_model.predict(new_test_X), axis = 1)
                right_count += sum(test_y == predict_y)
                print 'right', right_count, 'all', all_count, float(right_count)/all_count 
            have_load += 1        