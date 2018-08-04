import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from library.preprocess import load_data, IMG_HEIGHT, IMG_WIDTH
from library.hlac_features_calc import hlac_features_calc
from multiprocessing import Pool
from numpy import linalg as LA

processes = 8
hlac_dim = 25
train, test, test_label = load_data()
train = train.astype('float32') / 255.
test = test.astype('float32') / 255.

train = train.reshape(len(train), IMG_HEIGHT, IMG_WIDTH)
test = test.reshape(len(test), IMG_HEIGHT, IMG_WIDTH)

if __name__ == "__main__":
    hlac_train = Pool(processes=processes).map(hlac_features_calc, train)
    hlac_test = Pool(processes=processes).map(hlac_features_calc, test)
    hlac_test = np.asarray(hlac_test)
    X = np.asmatrix(hlac_train)
    print(X.shape)

    # do PCA by myself
    S = X.std(0)
    U = X.mean(0)
    X_norm = (X - U)/S
    C = X_norm.transpose().dot(X_norm) / (len(hlac_train) - 1)

    e_values, e_vectors = LA.eig(C)


    TH = 0.99999
    K = 0
    n_K = 0.
    sum_ev = np.sum(e_values)
    sum = 0.
    while n_K < TH:
        sum += e_values[K]
        n_K = sum / sum_ev
        K += 1

    print("new dimension: ", K)

    U_K = e_vectors[0:K]
    normal_space = np.transpose(U_K).dot(U_K)

    # calculate threshold = mean of anomaly_degree on training set
    anomaly_degree_train = np.zeros(len(hlac_train))
    i = 0
    for x in hlac_train:
        # x = (x - U) / S
        anomaly_degree = x.dot(normal_space).dot(np.transpose(x))
        anomaly_degree_train[i] = anomaly_degree
        i += 1
    threshold = np.mean(anomaly_degree_train)
    # threshold = np. max(anomaly_degree_train)
    print("Threshold: ", threshold)


    # test
    # accuracy estimation
    '''
    * to calculate precision, recall(sensitivity), accuracy, harmonic mean (f1 score)
    * precision = tp / (tp + fp)
    * recall = tp / (tp + fn)
    * accuracy = (tp + tn) / (tp + tn + fp + fn)
    * f1_score = 2 * precision * recall / (precision + recall)
    '''
    tp, fp, tn, fn = 0., 0., 0., 0.

    # anomaly_degree < threshold --> normal; anomaly_degree > threshold --> anomaly
    i = 0
    for x in hlac_test:
        # x = (x - U) / S
        anomaly_degree = x.dot(normal_space).dot(np.transpose(x))
        if (anomaly_degree < threshold):  # --> normal
            print(test_label[i] + ': ' + str(anomaly_degree) + ' --> normal')
            if (test_label[i] == 'normal'):  # true negative
                tn += 1
            else:  # false negative
                fn += 1
        else:  # --> anomaly
            print(test_label[i] + ': ' + str(anomaly_degree) + ' --> anomaly')
            if (test_label[i] == 'normal'):  # false positive
                fp += 1
            else:  # true positive
                tp += 1
        i += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('\nPrecesion: %.3f' % precision)
    print('\nRecall: %.3f' % recall)
    print('\nAccuracy: %.3f' % ((tp + tn) / (tp + tn + fn + fp)))
    print('\nHarmonic mean (f1_score): %.3f' % ((2 * precision * recall) / (precision + recall)))










