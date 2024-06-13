import glob

import numpy as np
import cv2

from numpy._typing import NDArray

from torch.utils.data import Dataset
from pandas.core.common import flatten

import torch
from torch import nn

# from multiprocessing import Process, Queue


# def load_dataset(dataset_path: str, process_id: int, num_workers: int, result: Queue):
def load_dataset(dataset_path: str):
    image_paths, labels = [], []
    for dataset in glob.glob(dataset_path + "/*"):
        label = dataset.split("/")[-1]

        if label in ("Mountain", "Other", "readme.txt"):
            continue

        labels.append(label)

        image_paths_for_label = glob.glob(dataset + "/*")
        image_paths.append(image_paths_for_label)
    image_paths = list(flatten(image_paths))

    idx_to_label = {i: j for i, j in enumerate(labels)}
    label_to_idx = {value: key for key, value in idx_to_label.items()}

    features, labels = [], []
    """
    for i, image_path in enumerate(image_paths):
        idx_range_start = (len(image_paths) // num_workers) * process_id
        idx_range_end = (len(image_paths) // num_workers) * (process_id + 1)
        if not (i < idx_range_start or i >= idx_range_end):
            continue
            """
    for image_path in image_paths:
        cur_image_path = image_path

        rgb_image = cv2.imread(cur_image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        hist_h, hist_s, hist_v = get_hsv_histogram(rgb_image)
        lbp_feature = get_lbp_feature(gray_image)
        glcm_1, glcm_2, glcm_3 = get_glcm_feature(gray_image)
        laws_texture_feature = get_laws_texture_feature(gray_image)

        label = cur_image_path.split("/")[-2]
        label_idx = label_to_idx[label]

        data = np.array(
            (
                hist_h,
                hist_s,
                hist_v,
                lbp_feature,
                glcm_1,
                glcm_2,
                glcm_3,
                laws_texture_feature,
            )
        )
        data = data.T
        features.append(data)
        labels.append(label_idx)
    features = np.array(features)
    labels = np.array(labels)

    result.put((features, labels))

    return


class RecaptchaDataset(Dataset):
    def __init__(self, dataset_path: str):
        super(RecaptchaDataset, self).__init__()

        image_paths, labels = [], []
        for dataset in glob.glob(dataset_path + "/*"):
            label = dataset.split("/")[-1]

            if label in ("Mountain", "Other", "readme.txt"):
                continue

            labels.append(label)

            image_paths_for_label = glob.glob(dataset + "/*")
            image_paths.append(image_paths_for_label)
        self.__image_paths = list(flatten(image_paths))
        self.__size = len(self.__image_paths)

        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}

    def __len__(self):
        return self.__size

    def __getitem__(self, idx):
        cur_image_path = self.__image_paths[idx]

        rgb_image = cv2.imread(cur_image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        hist_h, hist_s, hist_v = get_hsv_histogram(rgb_image)
        lbp_feature = get_lbp_feature(gray_image)
        glcm_1, glcm_2, glcm_3 = get_glcm_feature(gray_image)
        laws_texture_feature = get_laws_texture_feature(gray_image)

        label = cur_image_path.split("/")[-2]
        label_idx = self.label_to_idx[label]

        data = np.array(
            (
                hist_h,
                hist_s,
                hist_v,
                lbp_feature,
                glcm_1,
                glcm_2,
                glcm_3,
                laws_texture_feature,
            )
        )
        # data = data.T  # only used in LinearModel()

        return data, label_idx


def load_test_dataset(dataset_path: str):
    labels, image_paths = [], []
    for dataset in glob.glob(dataset_path + "/*"):
        label = dataset.split("/")[-1]
        labels.append(label)

        image_paths_for_label = glob.glob(dataset + "/*")
        image_paths.append(image_paths_for_label)
    image_paths = list(flatten(image_paths))

    idx_to_label = {i: j for i, j in enumerate(labels)}
    label_to_idx = {value: key for key, value in idx_to_label.items()}

    features, labels = [], []
    for image_path in image_paths:
        cur_image_path = image_path

        rgb_image = cv2.imread(cur_image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        hist_h, hist_s, hist_v = get_hsv_histogram(rgb_image)
        lbp_feature = get_lbp_feature(gray_image)
        glcm_1, glcm_2, glcm_3 = get_glcm_feature(gray_image)
        laws_texture_feature = get_laws_texture_feature(gray_image)

        label = cur_image_path.split("/")[-2]
        label_idx = label_to_idx[label]

        data = np.array(
            (
                hist_h,
                hist_s,
                hist_v,
                lbp_feature,
                glcm_1,
                glcm_2,
                glcm_3,
                laws_texture_feature,
            )
        )
        data = data.T
        features.append(data)
        labels.append(label_idx)
    features = np.array(features)
    labels = np.array(labels)

    return features, labels


class TestDataset(Dataset):
    def __init__(self, dataset_path: str):
        super(TestDataset, self).__init__()

        labels, image_paths = [], []
        for dataset in glob.glob(dataset_path + "/*"):
            label = dataset.split("/")[-1]
            labels.append(label)

            image_paths_for_label = glob.glob(dataset + "/*")
            image_paths.append(image_paths_for_label)
        self.__image_paths = list(flatten(image_paths))

        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}

    def __len__(self):
        return len(self.__image_paths)

    def __getitem__(self, idx):
        cur_image_path = self.__image_paths[idx]

        rgb_image = cv2.imread(cur_image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        hist_h, hist_s, hist_v = get_hsv_histogram(rgb_image)
        lbp_feature = get_lbp_feature(gray_image)
        glcm_1, glcm_2, glcm_3 = get_glcm_feature(gray_image)
        laws_texture_feature = get_laws_texture_feature(gray_image)

        label = cur_image_path.split("/")[-2]
        label_idx = self.label_to_idx[label]

        data = np.array(
            (
                hist_h,
                hist_s,
                hist_v,
                lbp_feature,
                glcm_1,
                glcm_2,
                glcm_3,
                laws_texture_feature,
            )
        )
        # data = data.T  # only used in LinearModel()

        return data, label_idx


def histogram_normalization(histogram: NDArray):
    histogram = histogram.astype("float")
    histogram /= histogram.sum()

    return histogram


def get_hsv_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    hist_h, bins_h = np.histogram(h, bins=10, range=(0, 256))
    hist_h = histogram_normalization(hist_h)

    hist_s, bins_s = np.histogram(s, bins=10, range=(0, 256))
    hist_s = histogram_normalization(hist_s)

    hist_v, bins_v = np.histogram(v, bins=10, range=(0, 256))
    hist_v = histogram_normalization(hist_v)

    # (3, 10)
    return hist_h, hist_s, hist_v


from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def get_lbp_feature(gray_image):
    lbp = local_binary_pattern(gray_image, P=8, R=1)
    hist_lbp, bin_lbp = np.histogram(lbp.ravel(), bins=10, range=(0, 256))
    hist_lbp = histogram_normalization(hist_lbp)

    # (1, 10)
    return hist_lbp


def get_glcm_feature(gray_image):
    glcm_features = []
    for i in range(0, 3):
        angle = (np.pi / 4) * i

        glcm = graycomatrix(
            gray_image,
            distances=[1, 2],
            angles=[angle],
            levels=256,
            symmetric=False,
            normed=True,
        )

        (
            max_prob,
            contrast_1,
            dissimilarity_1,
            homogeneity_1,
            energy_1,
            correlation_1,
            contrast_2,
            # dissimilarity_2,
            homogeneity_2,
            energy_2,
            correlation_2,
        ) = (
            np.max(glcm),
            graycoprops(glcm, "contrast")[0][0],
            graycoprops(glcm, "dissimilarity")[0][0],
            graycoprops(glcm, "homogeneity")[0][0],
            graycoprops(glcm, "energy")[0][0],
            graycoprops(glcm, "correlation")[0][0],
            graycoprops(glcm, "contrast")[1][0],
            # graycoprops(glcm, "dissimilarity")[1][0],
            graycoprops(glcm, "homogeneity")[1][0],
            graycoprops(glcm, "energy")[1][0],
            graycoprops(glcm, "correlation")[1][0],
        )  # all numpy arrays

        glcm_features.append(
            [
                max_prob,
                contrast_1,
                dissimilarity_1,
                homogeneity_1,
                energy_1,
                correlation_1,
                contrast_2,
                # dissimilarity_2,
                homogeneity_2,
                energy_2,
                correlation_2,
            ]
        )

    glcm_features = np.array(glcm_features)
    # (3, 10)
    return glcm_features


from scipy import signal as sg


def get_laws_texture_feature(gray_image):
    (rows, cols) = gray_image.shape[:2]

    smooth_kernel = (1 / 25) * np.ones((5, 5))
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")
    gray_processed = np.abs(gray_image - gray_smooth)

    filter_vectors = np.array(
        [
            [1, 4, 6, 4, 1],  # L5
            [-1, -2, 0, 2, 1],  # E5
            [-1, 0, 2, 0, 1],  # S5
            [1, -4, 6, -4, 1],
        ]
    )  # R5

    # 0:L5L5, 1:L5E5, 2:L5S5, 3:L5R5,
    # 4:E5L5, 5:E5E5, 6:E5S5, 7:E5R5,
    # 8:S5L5, 9:S5E5, 10:S5S5, 11:S5R5,
    # 12:R5L5, 13:R5E5, 14:R5S5, 15:R5R5
    filters = list()
    for i in range(4):
        for j in range(4):
            filters.append(
                np.matmul(
                    filter_vectors[i][:].reshape(5, 1),
                    filter_vectors[j][:].reshape(1, 5),
                )
            )

    conv_maps = np.zeros((rows, cols, 16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], "same")

    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1] + conv_maps[:, :, 4]) // 2)  # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2] + conv_maps[:, :, 8]) // 2)  # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3] + conv_maps[:, :, 12]) // 2)  # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7] + conv_maps[:, :, 13]) // 2)  # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6] + conv_maps[:, :, 9]) // 2)  # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11] + conv_maps[:, :, 14]) // 2)  # S5R5 / R5S5
    texture_maps.append(conv_maps[:, :, 10])  # S5S5
    texture_maps.append(conv_maps[:, :, 5])  # E5E5
    texture_maps.append(conv_maps[:, :, 15])  # R5R5
    texture_maps.append(conv_maps[:, :, 0])  # L5L5 (use to norm TEM)

    TEM = []
    """
    L5L5 = np.abs(texture_maps[9]).sum()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i].sum() / L5L5))
        """
    for i in range(10):
        TEM.append(np.abs(texture_maps[i].sum()))
    TEM = np.array(TEM)

    # (1, 10)
    return TEM


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(8, 1, dtype=torch.float64, device=device)

    def forward(self, x):
        out = self.linear(x)
        return out


def get_feature_weights(dataset):
    """
    result = Queue()
    num_workers, procs = 8, []
    for i in range(num_workers):
        p = Process(
            target=load_dataset, args=("./recaptcha-dataset", i, num_workers, result)
        )
        p.start()
        procs.append(p)

    datasets = []
    while not result.empty():
        dataset = result.get()
        datasets.append(dataset)

    pdb.set_trace()

    for p in procs:
        p.join()


    result = Queue()
    num_workers, procs = 8, []
    for i in range(num_workers):
        p = Process(
            target=load_dataset, args=("./test-dataset", i, num_workers, result)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    temp_dataset = result.get()
    temp_dataset = TensorDataset(torch.Tensor(temp_dataset))
    """
    temp_dataset = TestDataset("./test-dataset")
    from torch.utils import data

    train_dataloader = data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=8
    )
    validate_dataloader = data.DataLoader(
        temp_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    model = LinearModel()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4
    )

    from tqdm import tqdm

    best_epoch, best_validate_loss, best_model_weights = 0, float("inf"), (0, 0)
    for epoch in range(1, 10 + 1):
        model.train()
        correct, train_epoch_loss = 0, 0.0
        for batch in tqdm(train_dataloader):
            X, y = tuple(t.to(device) for t in batch)

            output = model(X).squeeze()
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * y.size(0)
            predicts = torch.argmax(output, dim=1)
            correct += (y == predicts).sum().float()

        train_epoch_loss = train_epoch_loss / len(train_dataloader.dataset)
        train_accuracy = correct / len(train_dataloader.dataset) * 100

        model.eval()
        validate_epoch_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(validate_dataloader):
                X, y = tuple(t.to(device) for t in batch)

                outputs = model(X).squeeze()
                loss = loss_fn(outputs, y)

                validate_epoch_loss += loss.item() * y.size(0)
                predicts = torch.argmax(outputs, dim=1)
                correct += (y == predicts).sum().float()

        validate_epoch_loss = validate_epoch_loss / len(validate_dataloader.dataset)
        validate_accuracy = correct / len(validate_dataloader.dataset) * 100

        print(
            f"Epoch {epoch}\t| Train Loss {train_epoch_loss:.4f}\tTrain Accuracy {train_accuracy:.4f}\tValidate Loss {validate_epoch_loss:.4f}\tValidate Accuracy {validate_accuracy:.4f}"
        )

        if validate_epoch_loss < best_validate_loss:
            best_epoch, best_validate_loss = epoch, validate_epoch_loss
            best_model_weights = (model.linear.weight, model.linear.bias)

    print(
        f"Best Validate Loss: {best_validate_loss} @ {best_epoch} with weights: {best_model_weights[0]}, bias: {best_model_weights[1]}"
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    # hist_h, hist_s, hist_v, lbp_feature, glcm_1, glcm_2, glcm_3, laws_texture_feature
    dataset = RecaptchaDataset("./recaptcha-dataset")
    # get_feature_weights(dataset) # to get X_weights

    from torch.utils import data

    train_dataloader = data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=8
    )

    from tqdm import tqdm

    X_features, y_s = [], []
    for batch in tqdm(train_dataloader):
        X, y = batch
        X_features.append(X)
        y_s.append(y)

    X_features = np.array(torch.cat(X_features))
    y_s = np.array(torch.cat(y_s))

    X_weights, X_bias = (
        np.array(
            [
                5.1377e-01,
                8.0769e-01,
                -3.7637e-01,
                6.6119e-01,
                2.1999e-01,
                -1.5192e-01,
                -6.1256e-02,
                -1.6483e-04,
            ]
        ),
        -0.1139,
    )
    X_features = np.sum(X_features * X_weights[:, np.newaxis], axis=1) + X_bias

    from sklearn import svm
    from sklearn import tree
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier

    clf_s = [
        svm.SVC(kernel="rbf"),
        tree.DecisionTreeClassifier(),
        LogisticRegression(random_state=42),
        RandomForestClassifier(n_estimators=50, random_state=42),
        GaussianNB(),
    ]

    import pickle

    estimators = []
    for i, clf in enumerate(clf_s):
        clf.fit(X_features, y_s)

        file_name = "clf_{}.pk".format(i)
        pickle.dump(clf, open(file_name, "wb"))
        estimators.append((file_name, clf))
    voting_ensemble = VotingClassifier(estimators=estimators, voting="hard")
    voting_ensemble.fit(X_features, y_s)
    pickle.dump(voting_ensemble, open("voting_ensemble.pk", "wb"))

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_features, y_s)
    pickle.dump(knn, open("knn.pk", "wb"))

    with open("X_features.npy", "wb") as file:
        np.save(file, X_features)

    labels = list(dataset.idx_to_label.values())
    labels = np.array(labels)
    with open("labels.npy", "wb") as file:
        np.save(file, labels)

    with open("X_labels.npy", "wb") as file:
        np.save(file, y_s)
