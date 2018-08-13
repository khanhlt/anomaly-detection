from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

my_path = os.path.abspath(os.path.dirname(__file__))
TRAIN_FOLDER = os.path.join(my_path, "../dataset/train")
TEST_NORM_FOLDER = os.path.join(my_path, "../dataset/test/normal")
TEST_ABNORM_FOLDER = os.path.join(my_path, "../dataset/test/anomaly")


def count_images(folder):
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])


def load_images(folder):
    res = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), grayscale=True)
        img = img_to_array(img)
        res.append(img)
    return res

def load_data():
    train = load_images(TRAIN_FOLDER)
    test_normal = load_images(TEST_NORM_FOLDER)
    test_anomaly = load_images(TEST_ABNORM_FOLDER)
    test = []
    for x in test_normal:
        test.append(x)
    for x in test_anomaly:
        test.append(x)

    test_label = []
    for i in range(len(test_normal)):
        test_label.append("normal")
    for i in range(len(test_anomaly)):
        test_label.append("anomaly")
    return train, test, test_label

# test
if __name__ == "__main__":
    train, test, test_label = load_data()
    print(train.shape)
    print(test.shape)
    print(test_label)



