from argparse import ArgumentParser


def get_train_parsed_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--cuda",
        default=True,
        type=bool,
        help="a boolean value whether you use cuda devices or not (default: True)",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="whether train or test (default: true - train)",
    )
    parser.add_argument(
        "--seed",
        metavar="S",
        type=int,
        default=42,
        help="Random Seed (default: 42)",
    )
    parser.add_argument(
        "--vit-pretrained-model",
        default="google/vit-large-patch32-384",
        type=str,
        help="Pretraining model that you want to get",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/recaptcha-dataset",
        help="The path of the train dataset (default: ./datasets/recaptcha-dataset)",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        default="./datasets/test-dataset",
        help="The path to the test dataset (default: ./datasets/test-dataset)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="input batch size (default: 32)",
    )
    parser.add_argument(
        "--number-workers",
        metavar="N",
        type=int,
        default=8,
        help="The number of dataloader workers (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="The number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        metavar="L",
        type=float,
        default=3e-4,
        help="The float value to train (default: 3e-4)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./saved_models/target_model.pt",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="efficientnet",
        help="model type (efficientnet | resnet152 | resnet50)",
    )

    return parser.parse_args()
