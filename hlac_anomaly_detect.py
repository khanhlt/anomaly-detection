import numpy as np
from library.hlac_features_calc import hlac_features_calc
from library.preprocess import load_data
from numpy import linalg as LA
from multiprocessing import Pool

hlac_dim = 25

if __name__ == "__main__":
    train, test, test_label = load_data()

    hlac_train = Pool(processes=5).map(hlac_features_calc, train)
    hlac_train = np.asarray(hlac_train)
    hlac_train = hlac_train.reshape(len(hlac_train), hlac_dim)

    hlac_test = Pool(processes=8).map(hlac_features_calc, test)
    hlac_test = np.asarray(hlac_test)
    hlac_test = hlac_test.reshape(len(hlac_test), hlac_dim)

    S = hlac_train.std(0)
    U = hlac_train.mean(0)
    X_norm = (hlac_train - U) / S  # normalize X
    C = X_norm.transpose().dot(X_norm) / (len(hlac_train) - 1)
    e_values, e_vectors = LA.eig(C)  # eigenvalues & eigenvectors of matrix C

    print("Eigenvalues: ", e_values)

    TH = 0.999  # threshold to calculate K - dimension after compress (TH < 1, usually set as 0.99, 0.999,...)
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
    normal_space = np.identity(hlac_dim) - np.transpose(U_K).dot(U_K)
    print("normal space: ", normal_space.shape)
    # calculate threshold = mean of anomaly_degree on training set
    # anomaly_degree_train = np.zeros(len(hlac_train))
    anomaly_degree_train = []
    for x in hlac_train:
        x = (x - U) / S
        anomaly_degree = np.prod(x.dot(normal_space).dot(np.transpose(x)))
        anomaly_degree_train.append(anomaly_degree)
    threshold = np.mean(anomaly_degree_train) + np.std(anomaly_degree_train)
    print("Threshold (mean): ", threshold)
    print("Max: ", np.max(anomaly_degree_train))
    print("Min: ", np.min(anomaly_degree_train))
    print("Std: ", np.std(anomaly_degree_train))


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

    # anomaly_degree < threshold --> normal
    # anomaly_degree > threshold --> anomaly
    i = 0
    for x in hlac_test:
        x = (x - U) / S
        anomaly_degree = np.prod(x.dot(normal_space).dot(np.transpose(x)))
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






