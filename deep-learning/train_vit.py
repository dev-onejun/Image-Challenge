import os
from datetime import datetime
from utils.args_parser import get_train_parsed_arguments

import torch
from torch.utils import data

from transformers import ViTImageProcessor
from torch import nn
from models import ViTClassifier

from datasets import RecaptchaDataset as Dataset

# from datasets import Cifar10 as Dataset
from utils.train_helpers import fit, evaluate

from torch.utils.tensorboard.writer import SummaryWriter


def train(train_loader, validate_loader, processor, model, n_epochs, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    """
    loss_fn = nn.CrossEntropyLoss()
    # scheduler

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = fit(
            processor,
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )
        validate_loss, validate_accuracy = evaluate(
            processor,
            model,
            validate_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch}\t|\tTrain Loss {train_loss:.4f}\tTrain Accuracy {train_accuracy:.4f}\tValidate Loss {validate_loss:.4f}\tValidate Accuracy {validate_accuracy:.4f}"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)
        writer.add_scalar("loss/validate", validate_loss, epoch)
        writer.add_scalar("accuracy/validate", validate_accuracy, epoch)

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(saved_model_dir, "epoch_{}.pt".format(epoch)),
            )

    writer.flush()

    torch.save(
        model.state_dict(),
        os.path.join(saved_model_dir, "final_{}.pt".format(datetime.now())),
    )


def test(test_loader, processor, model):
    loss_fn = nn.CrossEntropyLoss()

    _, test_accuracy = evaluate(processor, model, test_loader, loss_fn, device)
    print(f"Test Accuracy {test_accuracy:.4f}")


def main():
    processor = ViTImageProcessor.from_pretrained(args.vit_pretrained_model)
    model = ViTClassifier.from_pretrained(args.vit_pretrained_model)

    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if args.train:
        train_dataset = Dataset(args.dataset_path, "train")
        validate_dataset = Dataset(args.dataset_path, "validate")
        # train_dataset = Dataset(args.dataset_path, 'train', processor)

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
            processor,
            model,
            n_epochs=args.epochs,
            lr=args.learning_rate,
        )
    else:
        assert os.path.exists(
            args.model_path
        ), "A model file to test should be given with the --model-path argument"
        model.load_state_dict(torch.load(args.model_path))

        test_dataset = Dataset(args.dataset_path, "test")
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.number_workers,
        )

        test(test_loader, processor, model)

    writer.close()


writer = SummaryWriter()

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
    """A ViT Training Script for Computer Vision Challenges"""

    import sys, warnings

    warnings.filterwarnings("ignore")

    # sys.path.insert(0, os.getcwd())

    main()
