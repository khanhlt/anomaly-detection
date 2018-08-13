import numpy as np


def hlac_features_calc(img):
    window_size_3 = 3

    # filter pattern for HLAC (N = 0,1,2; window_size = 3 * 3)
    filter_list_3 = [
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]]),
    ]

    height = img.shape[0]
    width = img.shape[1]

    num_filter_3 = len(filter_list_3)
    features = np.zeros(num_filter_3)

    print('Calculating HLAC features...')

    for i in range(num_filter_3):
        filter = filter_list_3[i]
        for y in range(0, height - window_size_3 + 1):
            for x in range(0, width - window_size_3 + 1):
                local_area = img[y: y + window_size_3, x: x + window_size_3]  # area of image correspond to filter
                features[i] += np.prod(local_area[np.nonzero(filter)])

    return features


# test
if __name__ == "__main__":
    import os

    my_path = os.path.abspath(os.path.dirname(__file__))
    img_path_1 = os.path.join(my_path, "../hlac_test_data/testdata.png")

    from keras.preprocessing.image import load_img, img_to_array

    img = load_img(img_path_1, grayscale=True, target_size=(600, 600))
    img_arr = img_to_array(img)
    img_arr = img_arr.reshape(600, 600)

    img_arr_1 = img_arr[0:300, 0:300]
    img_arr_2 = img_arr[0:300, 300:600]
    img_arr_3 = img_arr[300:600, 0:300]
    img_arr_4 = img_arr[300:600, 300:600]

    features_1 = hlac_features_calc(img_arr_1)
    features_2 = hlac_features_calc(img_arr_2)
    features_3 = hlac_features_calc(img_arr_3)
    features_4 = hlac_features_calc(img_arr_4)

    a = features_1 - features_2
    b = features_3 + features_4 - features_2

    print(a)
    print('\n', b)
    print(features_1)
    print(features_2)
