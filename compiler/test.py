import numpy as np
import matplotlib.pyplot as plt
import random

image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "/Users/rohitgupta/Desktop/UCSD/quarter_2/cse_240d/mini_projects/autoencoder_mnist/"
train_data_d = np.loadtxt(data_path + "mnist_train.csv",delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

import pickle

# parts_to_split = 8;
# train_data_d = np.ones((120000, 785))

# start = 0
# end = 15000
# for a in np.arange(0, parts_to_split):
#     with open(data_path + "./pickled_mnist_train_" + str(a) + ".pkl", "br") as fh:
#         data = pickle.load(fh)
#     train_data_d[start:end, ] *= np.array(data[0])
#     start = end
#     end += 15000
#
# # train_data_d = np.array(train_data_d)
# # train_data_d = train_data_d.reshape(120000,785)
# print("read train_data_d read")
#
# with open(data_path + "./pickled_mnist_test.pkl", "br") as fh:
#     data = pickle.load(fh)
# test_data = data[0]
print("read test_data read")

image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


def mean_l2_loss(y_true, y_pred):
    return K.square(y_pred - y_true)


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

# 1
x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
print("1:conv2d", x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print("1:maxpooling", x.shape)

# 2
x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
print("2:conv2d", x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print("2:maxpooling", x.shape)

# 3
x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
print("3:conv2d", x.shape)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print("3:maxpooling", encoded.shape)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

# 4
x = Conv2D(8, (4, 4), activation='relu', padding='same')(encoded)
print("4:conv2d", x.shape)
x = UpSampling2D((2, 2))(x)
print("4:upsampling", x.shape)

# 5
x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
print("5:conv2d", x.shape)
x = UpSampling2D((2, 2))(x)
print("5:upsampling", x.shape)

# 6
x = Conv2D(16, (3, 3), activation='relu')(x)
print("6:conv2d", x.shape)
x = UpSampling2D((2, 2))(x)
print("6:upsampling", x.shape)

# 7
decoded = Conv2D(1, (4, 4), activation='sigmoid', padding='same')(x)
print("7:conv2d", decoded.shape)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss=mean_l2_loss)

from keras.datasets import mnist
import numpy as np

x_train = train_data_d[:, 1:].reshape(60000, 28, 28)
x_test = test_data[:, 1:].reshape(10000, 28, 28)
# (x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
print(x_train.shape)

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=30,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)


def calc_mean_l2_error(actual_inp, predicted_inp):
    a_num_img, a_x_dim, a_y_dim, a_temp1 = actual_inp.shape
    p_num_img, p_x_dim, p_y_dim, p_temp1 = predicted_inp.shape
    assert a_num_img == p_num_img
    assert a_x_dim == p_x_dim
    assert a_y_dim == p_y_dim
    assert a_temp1 == p_temp1
    # print(actual_inp.shape, predicted_inp.shape)
    temp_a = (predicted_inp.reshape(a_num_img, a_x_dim, a_y_dim)).flatten() \
             - (actual_inp.reshape(a_num_img, a_x_dim, a_y_dim)).flatten()
    # print(temp_a.shape)
    temp_a = np.square(temp_a)
    temp_a = temp_a.reshape(a_num_img, a_x_dim * a_y_dim)
    # print(temp_a.shape)
    temp_a = (temp_a.sum(axis=1)).flatten()
    temp_a = temp_a / 784
    return np.array(temp_a)


test_error = calc_mean_l2_error(x_test, decoded_imgs)
print("min_error", min(test_error))
min_error_index = np.where(test_error == min(test_error))

print("max_error", max(test_error))
max_error_index = np.where(test_error == max(test_error))

n = []
n.append(min_error_index)
n.append(max_error_index)

plt.figure(figsize=(20, 4))
for i in np.arange(1, len(n) + 1):
    # display original
    ax = plt.subplot(2, len(n), i)
    plt.imshow(x_test[n[i - 1]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # print(y_test[i,0])

    # display reconstruction
    ax = plt.subplot(2, len(n), i + len(n))
    plt.imshow(decoded_imgs[n[i - 1]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig('best_and_worst.png', bbox_inches='tight')

# from train data
decoded_imgs = autoencoder.predict(x_train)

test_error = calc_mean_l2_error(x_train, decoded_imgs)
print("min_error", min(test_error))
min_error_index = np.where(test_error == min(test_error))

print("max_error", max(test_error))
max_error_index = np.where(test_error == max(test_error))

n = []
n.append(min_error_index)
n.append(max_error_index)

plt.figure(figsize=(20, 4))
for i in np.arange(1, len(n) + 1):
    # display original
    ax = plt.subplot(2, len(n), i)
    plt.imshow(x_train[n[i - 1]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # print(y_test[i,0])

    # display reconstruction
    ax = plt.subplot(2, len(n), i + len(n))
    plt.imshow(decoded_imgs[n[i - 1]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig('best_and_worst_train.png', bbox_inches='tight')
