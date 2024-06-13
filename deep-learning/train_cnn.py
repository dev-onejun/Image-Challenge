import os
from utils.args_parser import get_train_parsed_arguments

from models import EfficientNet, ResNet50, ResNet152
from models import EfficientNet_V2_L_Weights, ResNet152_Weights, ResNet50_Weights

from datasets import RecaptchaDataset as TrainDataset
from datasets import TestDataset
from torch.utils import data
import torch

from utils.train_helpers import fit_cnn, evaluate_cnn

from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from copy import deepcopy


def train(train_loader, validate_loader, model, n_epochs, lr):
    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_epoch, best_accuracy, best_model_weights = 0, 0.0, None
    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = fit_cnn(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )
        validate_loss, validate_accuracy = evaluate_cnn(
            model,
            validate_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch}\t| Train Loss {train_loss:.4f}\tTrain Accuracy {train_accuracy:.4f}\tValidate Loss {validate_loss:.4f}\tValidate Accuracy {validate_accuracy:.4f}"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)
        writer.add_scalar("loss/validate", validate_loss, epoch)
        writer.add_scalar("accuracy/validate", validate_accuracy, epoch)

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    saved_model_dir, "epoch_{}_{}.pt".format(epoch, datetime.now())
                ),
            )

        if validate_accuracy > best_accuracy:
            best_epoch, best_accuracy = epoch, validate_accuracy
            best_model_weights = deepcopy(model.state_dict())

    writer.flush()

    torch.save(
        model.state_dict(),
        os.path.join(saved_model_dir, "final_{}.pt".format(datetime.now())),
    )

    print(f"\n\nBest Accuracy: {best_accuracy}")
    torch.save(
        best_model_weights,
        os.path.join(
            saved_model_dir, "best_{}_{}.pt".format(best_epoch, datetime.now())
        ),
    )


def main():
    assert args.model_type in ("efficientnet", "resnet152", "resnet50")
    model, transform = (
        (EfficientNet(), EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms())
        if args.model_type == "efficientnet"
        else (
            (ResNet152(), ResNet152_Weights.IMAGENET1K_V2.transforms())
            if args.model_type == "resnet152"
            else (ResNet50(), ResNet50_Weights.IMAGENET1K_V2.transforms())
        )
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_dataset = TrainDataset(args.dataset_path, "all", transform=transform)
    validate_dataset = TestDataset(args.test_dataset_path, transform=transform)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.number_workers,
    )
    validate_loader = data.DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.number_workers,
    )
    train(
        train_loader,
        validate_loader,
        model,
        n_epochs=args.epochs,
        lr=args.learning_rate,
    )


args = get_train_parsed_arguments()

if args.cuda:
    assert (
        args.cuda == torch.cuda.is_available()
    ), "A CUDA Device is required to use cuda"
device = torch.device("cuda" if args.cuda else "cpu")


torch.manual_seed(args.seed)

saved_model_dir = "./saved_models"
os.makedirs(saved_model_dir, exist_ok=True)

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()
