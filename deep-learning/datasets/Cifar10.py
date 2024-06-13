from torch.utils.data import Dataset

from torchvision import datasets


class Cifar10(Dataset):
    def __init__(self, dataset_path, train_type):
        super(Cifar10, self).__init__()

        self.cifar10 = datasets.CIFAR10(dataset_path, train=True, download=True)
        self.total_size = len(self.cifar10.targets)

        assert train_type in (
            "train",
            "validate",
            "test",
        ), "The `train_type` variable must be gotten train, validate, or test"
        self.train_type = train_type

    def __len__(self):
        size = (
            int(self.total_size * 0.8)
            if self.train_type == "train"
            else int(self.total_size * 0.1)
        )

        return size

    def __getitem__(self, idx):
        train_idx = int(self.total_size * 0.8)
        validate_idx = train_idx + int(self.total_size * 0.1)
        test_idx = validate_idx + int(self.total_size * 0.1)

        image, target = (
            (self.cifar10.data[0 + idx], self.cifar10.targets[0 + idx])
            if self.train_type == "train"
            else (
                (
                    self.cifar10.data[train_idx + idx],
                    self.cifar10.targets[train_idx + idx],
                )
                if self.train_type == "validate"
                else (
                    self.cifar10.data[validate_idx + idx],
                    self.cifar10.targets[validate_idx + idx],
                )
            )
        )

        return image, target


if __name__ == "__main__":
    dataset = Cifar10("./cifar10-dataset", "train")

    from torch.utils import data

    dataloader = data.DataLoader(dataset)

    for i, data in enumerate(dataloader):
        print(i, end=" ")
