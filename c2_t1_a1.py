import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.models import (
    EfficientNet_V2_L_Weights,
    ResNet152_Weights,
    ResNet50_Weights,
)

import torch
from torch import nn
from CNNs import ResNet50, ResNet152, EfficientNet


class TestDataset(Dataset):
    def __init__(self, dataset_path, transform):
        super(TestDataset, self).__init__()

        image_paths = []
        for image_path in glob.glob(dataset_path + "/*"):
            image_paths.append(image_path)

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        image = self.transform(image)

        query_file_name = self.image_paths[idx].split("/")[-1]

        return image, query_file_name


class VotingEnsemble(nn.Module):
    def __init__(self, **kwargs):
        super(VotingEnsemble, self).__init__()

        self.resnet50 = ResNet50()
        self.resnet50.load_state_dict(torch.load(kwargs["resnet50"]))
        for name, param in self.resnet50.named_parameters():
            param.requires_grad = False

        self.resnet152 = ResNet152()
        self.resnet152.load_state_dict(torch.load(kwargs["resnet152"]))
        for name, param in self.resnet152.named_parameters():
            param.requires_grad = False

        self.efficientnetv2 = EfficientNet()
        self.efficientnetv2.load_state_dict(torch.load(kwargs["efficientnetv2"]))
        for name, param in self.efficientnetv2.named_parameters():
            param.requires_grad = False

    def forward(self, resnet50_input, resnet152_input, efficientnetv2_input):
        outputs = [
            self.resnet50(resnet50_input),
            self.resnet152(resnet152_input),
            self.efficientnetv2(efficientnetv2_input),
        ]  # (3, batch_size, num_classes)
        outputs = [torch.argmax(output, dim=1) for output in outputs]

        temp = outputs.copy()

        # Stack outputs to shape (num_models, batch_size, num_classes)
        outputs = torch.stack(outputs)

        mode_outputs, mode_indicies = torch.mode(outputs, dim=0)

        for i, mode_index in enumerate(mode_indicies):
            # Refers to the frequency of the classes is same
            if mode_index == 0:
                # Follows the highest accuracy model while training
                mode_outputs[i] = temp[0][i]

        return mode_outputs


def process_large_dataset(dataloader):
    for (resnet50, _), (resnet152, _), (efficientnetv2, query) in zip(
        dataloader["resnet50"],
        dataloader["resnet152"],
        dataloader["efficientnetv2"],
    ):
        yield (resnet50, resnet152, efficientnetv2, query)


labels = [
    "Car",
    "Motorcycle",
    "Traffic Light",
    "Bus",
    "Bicycle",
    "Hydrant",
    "Palm",
    "Chimney",
    "Bridge",
    "Crosswalk",
]
idx_to_label = {key: label for key, label in enumerate(labels)}

if __name__ == "__main__":
    import warnings, csv
    from torch.utils import data

    warnings.filterwarnings("ignore")

    DATASET_PATH = "./query"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = {
        "efficientnetv2": EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms(),
        "resnet50": ResNet50_Weights.IMAGENET1K_V2.transforms(),
        "resnet152": ResNet152_Weights.IMAGENET1K_V2.transforms(),
    }

    dataset = {}
    dataloader = {}
    for key, transform in transforms.items():
        dataset[key] = TestDataset(DATASET_PATH, transform=transform)
        dataloader[key] = data.DataLoader(
            dataset[key],
            batch_size=32,
            num_workers=8,
        )

    model = VotingEnsemble(
        resnet50="./saved_models/resnet50.pt",
        resnet152="./saved_models/resnet152.pt",
        efficientnetv2="./saved_models/efficientnetv2.pt",
    )
    model.to(device)

    answers = []
    model.eval()
    with torch.no_grad():
        for batch in process_large_dataset(dataloader):
            query = batch[3]
            batch = batch[:3]
            batch = tuple(t.to(device) for t in batch)
            resnet50_input, resnet152_input, efficientnetv2_input = batch

            outputs = model(resnet50_input, resnet152_input, efficientnetv2_input)

            for answer in zip(query, outputs):
                answers.append(answer)

    with open(__file__.split("/")[-1].split(".")[0] + ".csv", "w") as file:
        write = csv.writer(file)
        for answer in answers:
            query, idx = answer[0], answer[1].item()
            answer = query, idx_to_label[idx]

            write.writerow(answer)
