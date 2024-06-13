import numpy as np
import cv2

from numpy._typing import NDArray

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

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
    for i in range(10):
        TEM.append(np.abs(texture_maps[i].sum()))
    TEM = np.array(TEM)

    # (1, 10)
    return TEM
