from early_vision import (
    get_hsv_histogram,
    get_lbp_feature,
    get_glcm_feature,
    get_laws_texture_feature,
    X_weights,
    X_bias,
)

import glob, cv2, pickle, csv
import numpy as np

from sklearn.neighbors import KNeighborsClassifier


def load_test_dataset(dataset_path: str):
    image_paths, query_file_name = [], []
    for image_path in glob.glob(dataset_path + "/*"):
        image_paths.append(image_path)

        query_file_name.append(image_path.split("/")[-1])

    features = []
    for image_path in image_paths:
        cur_image_path = image_path

        rgb_image = cv2.imread(cur_image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        hist_h, hist_s, hist_v = get_hsv_histogram(rgb_image)
        lbp_feature = get_lbp_feature(gray_image)
        glcm_1, glcm_2, glcm_3 = get_glcm_feature(gray_image)
        laws_texture_feature = get_laws_texture_feature(gray_image)

        data = (
            hist_h,
            hist_s,
            hist_v,
            lbp_feature,
            glcm_1,
            glcm_2,
            glcm_3,
            laws_texture_feature,
        )
        features.append(data)

    features, query_file_name = np.array(features), np.array(query_file_name)

    return features, query_file_name


if __name__ == "__main__":
    with open("./DB/X_labels.npy", "rb") as file:
        X_labels = np.load(file)
    with open("./DB/labels.npy", "rb") as file:
        labels = np.load(file)

    query_X, query_file_names = load_test_dataset("./query")
    query_X = np.sum(query_X * X_weights[:, np.newaxis], axis=1) + X_bias

    model = pickle.load(open("./saved_models/knn.pk", "rb"))
    neigh_ind = model.kneighbors(X=query_X, n_neighbors=10, return_distance=False)

    neigh_labels = labels[X_labels[neigh_ind]]  # (100, 10)

    with open(__file__.split("/")[-1].split(".")[0] + ".csv", "w") as file:
        write = csv.writer(file)
        for query_file_name, prediction in zip(query_file_names, neigh_labels):
            write.writerow([query_file_name] + list(prediction))
