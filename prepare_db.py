from pandas.core.common import flatten
import torch
from torch import nn

import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    EfficientNet_V2_L_Weights,
    ResNet152_Weights,
    ResNet50_Weights,
)
from CNNs import ResNet50, ResNet152, EfficientNet

from tqdm import tqdm


class OriginalDataset(Dataset):
    def __init__(self, dataset_path, transform):
        super(OriginalDataset, self).__init__()

        labels = []
        image_paths = []
        for dataset in glob.glob(dataset_path + "/*"):
            label = dataset.split("/")[-1]

            if label in ("Mountain", "Other", "readme.txt"):
                continue

            labels.append(label)

            image_paths_for_label = glob.glob(dataset + "/*")
            image_paths.append(image_paths_for_label)

        self._image_paths = list(flatten(image_paths))
        self._size = len(self._image_paths)

        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}
        self.transform = transform

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx])
        image = image.convert("RGB")
        image = self.transform(image)

        label = self._image_paths[idx].split("/")[-2]
        # label_idx = self.label_to_idx[label]

        return image, label


class HiddenModels:
    def __init__(self, **kwargs):
        super(HiddenModels, self).__init__()

        assert tuple(kwargs.keys()) == ("resnet50", "resnet152", "efficientnetv2")

        _resnet50 = ResNet50()
        _resnet50.load_state_dict(torch.load(kwargs["resnet50"]))
        _resnet50.model.fc = nn.Identity()
        for name, param in _resnet50.named_parameters():
            param.requires_grad = False

        _resnet152 = ResNet152()
        _resnet152.load_state_dict(torch.load(kwargs["resnet152"]))
        _resnet152.model.fc = nn.Identity()
        for name, param in _resnet152.named_parameters():
            param.requires_grad = False

        _efficientnetv2 = EfficientNet()
        _efficientnetv2.load_state_dict(torch.load(kwargs["efficientnetv2"]))
        _efficientnetv2.model.classifier = nn.Identity()
        for name, param in _efficientnetv2.named_parameters():
            param.requires_grad = False

        self.models = {
            "resnet50": _resnet50,
            "resnet152": _resnet152,
            "efficientnetv2": _efficientnetv2,
        }

    def get_models(self):
        return self.models


def process_large_dataset(dataloader):
    for data in tqdm(dataloader):
        yield data


if __name__ == "__main__":
    import warnings, os
    from torch.utils import data

    warnings.filterwarnings("ignore")

    DATASET_PATH = "./recaptcha-dataset/Large"
    GENERATED_DB_PATH = "./DB"

    os.makedirs(GENERATED_DB_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = {
        "efficientnetv2": EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms(),
        "resnet50": ResNet50_Weights.IMAGENET1K_V2.transforms(),
        "resnet152": ResNet152_Weights.IMAGENET1K_V2.transforms(),
    }

    dataset = {}
    dataloader = {}
    for key, transform in transforms.items():
        dataset[key] = OriginalDataset(DATASET_PATH, transform=transform)
        dataloader[key] = data.DataLoader(
            dataset[key],
            batch_size=32,
            num_workers=8,
        )

    models = HiddenModels(
        resnet50="./saved_models/resnet50.pt",
        resnet152="./saved_models/resnet152.pt",
        efficientnetv2="./saved_models/efficientnetv2.pt",
    ).get_models()

    for transform, (model_type, model), dataloader in zip(
        transforms.values(), models.items(), dataloader.values()
    ):
        print(f"Generate DB for {model_type}")
        DB_FOR_MODEL = os.path.join(GENERATED_DB_PATH, model_type)
        os.makedirs(DB_FOR_MODEL, exist_ok=True)

        model.to(device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(process_large_dataset(dataloader)):
                input_images, labels = batch[0].to(device), batch[1]

                hidden_features = model(input_images)

                for hidden_feature, label in zip(hidden_features, labels):
                    FEATURE_PATH = os.path.join(DB_FOR_MODEL + f"/{label}_{i}.pt")
                    torch.save(hidden_feature, FEATURE_PATH)
