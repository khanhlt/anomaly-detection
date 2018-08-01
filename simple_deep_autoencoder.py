from library.preprocess import load_data, IMG_WIDTH, IMG_HEIGHT
from keras.layers import Input, Dense
from keras import Model
from sklearn.model_selection import train_test_split
import numpy as np

epochs = 30
train, test, test_label = load_data()
train = train.astype('float32') / 255.
test = test.astype('float32') / 255.

train = train.reshape((len(train), np.prod(train.shape[1:])))
test = test.reshape((len(test), np.prod(test.shape[1:])))

x_train, x_test = train_test_split(train, test_size=0.2, random_state=10)

img_shape = train[0].shape


# autoencoder layers
input_img = Input(shape=img_shape)
encoding_dim = IMG_WIDTH * IMG_HEIGHT  # 32 floats

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(encoding_dim, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=2)

'''
* threshold = max_loss on x_test
'''
threshold = 0
for x in x_test:
    x = np.expand_dims(x, axis=0)
    loss = autoencoder.test_on_batch(x, x)
    if threshold < loss:
        threshold = loss
print(threshold)

'''
* threshold = mean(x_test_loss)
* decrease accuracy but can detect almost anomalies
'''
# sum, i = 0., 0.
# for x in x_test:
#     x = np.expand_dims(x, axis=0)
#     loss = autoencoder.test_on_batch(x, x)
#     sum += loss
#     i += 1
# threshold = sum / i


'''
* to calculate precision, recall(sensitivity), & accuracy
* precision = tp / (tp + fp)
* recall = tp / (tp + fn)
* accuracy = (tp + tn) / (tp + tn + fp + fn)
'''
tp, fp, tn, fn = 0., 0., 0., 0.


# loss < threshold --> normal; loss > threshold --> anomaly
i = 0
for x in test:
    x = np.expand_dims(x, axis=0)
    loss = autoencoder.test_on_batch(x, x)
    if (loss < threshold):              # --> normal
        print('%s: %f --> normal' % (test_label[i], loss))
        if(test_label[i] == 'normal'):    # true negative
            tn += 1
        else:   # false negative
            fn += 1
    else:                               # --> anomaly
        print('%s: %f --> anomaly' % (test_label[i], loss))
        if(test_label[i] == 'normal'):    # false positive
            fp += 1
        else:   # true positive
            tp += 1
    i += 1


print('\nPrecesion: %.3f' % (tp/(tp+fp)))
print('\nRecall: %.3f' % (tp/(tp+fn)))
print('\nAccuracy: %.3f' % ((tp+tn)/(tp+tn+fn+fp)))



