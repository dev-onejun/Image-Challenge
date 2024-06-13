from tqdm import tqdm

import torch


def process_large_dataset(dataloader):
    for data in tqdm(dataloader):
        yield data


def fit(processor, model, dataloader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    correct = 0
    # for images, targets in tqdm(train_loader):
    for images, targets in process_large_dataset(dataloader):
        images, targets = images.to(device), targets.to(device)

        inputs = processor(images, do_rescale=False)
        inputs = torch.Tensor(inputs["pixel_values"]).to(device)
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        predicts = torch.argmax(outputs, dim=1)
        correct += (targets == predicts).sum().float()

        """
        batch_correct = (targets == predicts).sum().float()
        batch_accuracy = batch_correct / args.batch_size * 100
        print(f"Step Accuracy {batch_accuracy:.4f}")
        """

    epoch_loss = epoch_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset) * 100

    return epoch_loss, accuracy


def evaluate(processor, model, dataloader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    correct = 0
    with torch.no_grad():
        # for images, targets in tqdm(validate_loader):
        for images, targets in process_large_dataset(dataloader):
            images, targets = images.to(device), targets.to(device)

            inputs = processor(images, do_rescale=False)
            inputs = torch.Tensor(inputs["pixel_values"]).to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            epoch_loss += loss.item() * images.size(0)
            predicts = torch.argmax(outputs, dim=1)
            correct += (targets == predicts).sum().float()

    epoch_loss = epoch_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset) * 100

    return epoch_loss, accuracy


def fit_cnn(model, dataloader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    correct = 0
    for images, targets in process_large_dataset(dataloader):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        predicts = torch.argmax(outputs, dim=1)
        correct += (targets == predicts).sum().float()

    epoch_loss = epoch_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset) * 100

    return epoch_loss, accuracy


def evaluate_cnn(model, dataloader, loss_fn, device):
    model.eval()

    epoch_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, targets in process_large_dataset(dataloader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            # loss.item() = batch average loss && images.size(0) = batch size (can differ in the last)
            epoch_loss += loss.item() * images.size(0)
            predicts = torch.argmax(outputs, dim=1)
            correct += (targets == predicts).sum().float()

    epoch_loss = epoch_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset) * 100

    return epoch_loss, accuracy
