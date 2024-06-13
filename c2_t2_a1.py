import glob
from PIL import Image

from torch.utils.data import Dataset


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


import torch
from torch import nn
from CNNs import ResNet50, ResNet152, EfficientNet


class HiddenModels(nn.Module):
    def __init__(self, **kwargs):
        super(HiddenModels, self).__init__()

        self.resnet50 = ResNet50()
        self.resnet50.load_state_dict(torch.load(kwargs["resnet50"]))
        self.resnet50.model.fc = nn.Identity()
        for name, param in self.resnet50.named_parameters():
            param.requires_grad = False

        self.resnet152 = ResNet152()
        self.resnet152.load_state_dict(torch.load(kwargs["resnet152"]))
        self.resnet152.model.fc = nn.Identity()
        for name, param in self.resnet152.named_parameters():
            param.requires_grad = False

        self.efficientnetv2 = EfficientNet()
        self.efficientnetv2.load_state_dict(torch.load(kwargs["efficientnetv2"]))
        self.efficientnetv2.model.classifier = nn.Identity()
        for name, param in self.efficientnetv2.named_parameters():
            param.requires_grad = False

    def forward(self, resnet50_input, resnet152_input, efficientnetv2_input):
        hidden_features = (
            self.resnet50(resnet50_input),
            self.resnet152(resnet152_input),
            self.efficientnetv2(efficientnetv2_input),
        )

        return hidden_features


def process_large_dataset(dataloader):
    for (resnet50, _), (resnet152, _), (efficientnetv2, query) in zip(
        dataloader["resnet50"],
        dataloader["resnet152"],
        dataloader["efficientnetv2"],
    ):
        yield (resnet50, resnet152, efficientnetv2, query)


class DB(Dataset):
    def __init__(self):
        super(DB, self).__init__()

        self.db = []
        labels = []
        for model_type in glob.glob("./DB/*"):
            if model_type.split("/")[-1] not in (
                "efficientnetv2",
                "resnet152",
                "resnet50",
            ):
                continue

            db_per_model = []
            for tensor_file in glob.glob(model_type + "/*"):
                data = torch.load(tensor_file)
                db_per_model.append(data)

                if len(self.db) == 0:
                    label = tensor_file.split("/")[-1].split("_")[0]
                    labels.append(label)

            db_per_model = torch.stack(db_per_model)  # (image_#, hidden_size)
            self.db.append(db_per_model)

        self._size = len(self.db)
        self.idx_to_label = {i: j for i, j in enumerate(labels)}
        self.label_to_idx = {value: key for key, value in self.idx_to_label.items()}

    def __len__(self):
        # idx - 0: resnet50, 1: resnet152, 2: efficientnetv2
        return self._size

    def __getitem__(self, idx):
        return self.db[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    import pdb
    import warnings, csv
    import numpy as np
    from torch.utils import data

    warnings.filterwarnings("ignore")

    DATASET_PATH = "./query"

    from torchvision.models import (
        EfficientNet_V2_L_Weights,
        ResNet152_Weights,
        ResNet50_Weights,
    )

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

    model = HiddenModels(
        resnet50="./saved_models/resnet50.pt",
        resnet152="./saved_models/resnet152.pt",
        efficientnetv2="./saved_models/efficientnetv2.pt",
    )
    model.to(device)
    model.eval()

    db_all = DB()  # 0: resnet50, 1: resnet152, 2: efficientnetv2
    idx_to_label = db_all.idx_to_label

    answers, query_hidden_features, queries = [], [[], [], []], []
    with torch.no_grad():
        for batch in process_large_dataset(dataloader):
            data, query = batch[:3], batch[3]
            data = tuple(t.to(device) for t in data)
            resnet50_input, resnet152_input, efficientnetv2_input = data

            resnet50_hiddens, resnet152_hiddens, effv2_hiddens = model(
                resnet50_input, resnet152_input, efficientnetv2_input
            )  # 3 * (batch, hidden_feature): 3 * (batch, 2048|1280)

            query_hidden_features[0] += resnet50_hiddens.cpu()
            query_hidden_features[1] += resnet152_hiddens.cpu()
            query_hidden_features[2] += effv2_hiddens.cpu()
            queries += query
        # query_hidden_features: (3, 100, 2048|1280), query: (3, 100)

        sim_per_model = []
        for db, hidden_feature in zip(db_all, query_hidden_features):
            hidden_feature = np.array(hidden_feature)
            hidden_feature = torch.Tensor(hidden_feature).to(device)

            query_top10 = []
            for query_hidden_feature in hidden_feature:
                query_hidden_feature.unsqueeze(0)
                """ euclidian distance """
                distance = torch.norm(db - query_hidden_feature, dim=1, p=2)
                knn = distance.topk(10, largest=False)
                # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

                """ cosine similarity """
                # torch.cosine_similarity(db, query_hidden_feature, dim=)

                query_top10.append(knn.indices.cpu())
            # query_top10: (100, 10)
            sim_per_model.append(query_top10)
        # sim_per_model: (3, 100, 10)

        sim_per_model = np.array(sim_per_model)
        sim_per_model = torch.Tensor(sim_per_model).to(device)
        mode_outputs, mode_indicies = torch.mode(sim_per_model, dim=0)

        for query, mode_output in zip(queries, mode_outputs):
            top_k_prediction = []
            for top_k_idx in mode_output:
                top_k_idx = int(top_k_idx.item())
                top_k_prediction.append(idx_to_label[top_k_idx])
            answers.append(([query] + top_k_prediction))

    with open(__file__.split("/")[-1].split(".")[0] + ".csv", "w") as file:
        write = csv.writer(file)
        for answer in answers:
            """
            query, idx = answer[0], answer[1]
            answer = query, idx_to_label[idx]
            """

            write.writerow(answer)
