# https://towardsdatascience.com/custom-dataset-in-pytorch-part-1-images-2df3152895

from pandas.core.common import flatten

from torch.utils.data import Dataset
from torchvision import transforms

import glob
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ]
)


class RecaptchaDataset(Dataset):
    def __init__(self, dataset_path, train_type, transform=transform):
        super(RecaptchaDataset, self).__init__()

        labels = []
        train_image_paths, validate_image_paths, test_image_paths = [], [], []
        for dataset in glob.glob(dataset_path + "/*"):
            label = dataset.split("/")[-1]

            if label in ("Mountain", "Other", "readme.txt"):
                continue

            labels.append(label)

            image_paths_for_label = glob.glob(dataset + "/*")
            (
                train_image_paths_for_label,
                validate_image_paths_for_label,
                test_image_paths_for_label,
            ) = (
                image_paths_for_label[: int(0.8 * len(image_paths_for_label))],
                image_paths_for_label[
                    int(0.8 * len(image_paths_for_label)) : int(
                        0.9 * len(image_paths_for_label)
                    )
                ],
                image_paths_for_label[int(0.9 * len(image_paths_for_label)) :],
            )

            train_image_paths.append(train_image_paths_for_label)
            validate_image_paths.append(validate_image_paths_for_label)
            test_image_paths.append(test_image_paths_for_label)

        self.train_image_paths, self.validate_image_paths, self.test_image_paths = (
            list(flatten(train_image_paths)),
            list(flatten(validate_image_paths)),
            list(flatten(test_image_paths)),
        )
        """
        self.train_image_paths, self.validate_image_paths, self.test_image_paths = (
            random.shuffle(train_image_paths),
            random.shuffle(validate_image_paths),
            random.shuffle(test_image_paths),
        )
        """

        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}

        self.transform = transform

        assert train_type in (
            "train",
            "validate",
            "test",
            "all",
        ), "The `train_type` variable must be gotten train, validate, or test"

        self.train_type = train_type

    def __len__(self):
        size = (
            len(self.train_image_paths)
            if self.train_type == "train"
            else (
                len(self.validate_image_paths)
                if self.train_type == "validate"
                else (
                    len(self.test_image_paths)
                    if self.train_type == "test"
                    else len(self.train_image_paths)
                    + len(self.validate_image_paths)
                    + len(self.test_image_paths)
                )
            )
        )

        return size

    def __getitem__(self, idx):
        image_paths = (
            self.train_image_paths
            if self.train_type == "train"
            else (
                self.validate_image_paths
                if self.train_type == "validate"
                else (
                    self.test_image_paths
                    if self.train_type == "test"
                    else self.train_image_paths
                    + self.validate_image_paths
                    + self.test_image_paths
                )
            )
        )

        image = Image.open(image_paths[idx])
        image = image.convert("RGB")
        image = self.transform(image)

        label = image_paths[idx].split("/")[-2]
        label = self.label_to_idx[label]

        return image, label


class TestDataset(Dataset):
    def __init__(self, dataset_path, transform=transform):
        super(TestDataset, self).__init__()

        labels, image_paths = [], []
        for dataset in glob.glob(dataset_path + "/*"):
            label = dataset.split("/")[-1]
            labels.append(label)

            image_paths_for_label = glob.glob(dataset + "/*")
            image_paths.append(image_paths_for_label)

        self.image_paths = list(flatten(image_paths))
        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        image = self.transform(image)

        label = self.image_paths[idx].split("/")[-2]
        label = self.label_to_idx[label]

        return image, label


if __name__ == "__main__":
    train_dataset = RecaptchaDataset("./augmented-recaptcha-dataset", "all")
    print(len(train_dataset))
    print(train_dataset.idx_to_label)

    test_dataset = TestDataset("./test-dataset")
    print(len(test_dataset))
