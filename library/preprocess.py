from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

IMG_HEIGHT = 288
IMG_WIDTH = 432
IMG_HEIGHT_RESIZE, IMG_WIDTH_RESIZE = 300, 300
RGB_DEPTH = 3
GRAYSCALE_DEPTH = 1
my_path = os.path.abspath(os.path.dirname(__file__))
TRAIN_FOLDER = os.path.join(my_path, "../dataset/train")
TEST_FOLDER = os.path.join(my_path, "../dataset/test")


def count_images(folder):
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])


def load_images(folder):
    image_array = np.zeros((count_images(folder), IMG_HEIGHT, IMG_WIDTH, GRAYSCALE_DEPTH))
    i = 0
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), grayscale=True, target_size=(IMG_HEIGHT, IMG_WIDTH))
        arr = img_to_array(img)
        image_array[i] = arr
        i += 1
    return image_array

def load_images_resize(folder):
    image_array = np.zeros((count_images(folder), IMG_HEIGHT_RESIZE, IMG_WIDTH_RESIZE, GRAYSCALE_DEPTH))
    i = 0
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), grayscale=True, target_size=(IMG_HEIGHT_RESIZE, IMG_WIDTH_RESIZE))
        arr = img_to_array(img)
        image_array[i] = arr
        i += 1
    return image_array

def load_label(folder):
    label_array = []
    for filename in os.listdir(folder):
        if(filename[0] == 'n'):
            label_array.append('normal')
        if(filename[0] == 'a'):
            label_array.append('anomaly')
    return label_array


def load_data():
    train = load_images(TRAIN_FOLDER)
    test = load_images(TEST_FOLDER)
    test_label = load_label(TEST_FOLDER)

    return train, test, test_label

def load_data_resize():
    train = load_images_resize(TRAIN_FOLDER)
    test = load_images_resize(TEST_FOLDER)
    test_label = load_label(TEST_FOLDER)

    return train, test, test_label

# test
if __name__ == "__main__":
    train, test, test_label = load_data()
    print(train.shape)
    print(test.shape)
    print(test_label)



