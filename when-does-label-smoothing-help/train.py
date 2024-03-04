"""
Bare bones training script.

The pooling in ImageNet models may be inappropriate for Cifar10 as-is.
- ResNet50 and ResNeXt50 downsample input by 32.
- It is common to use 224/256 pixel crops for ImageNet.
- Cifar10 is made of 32 pixel chips.
"""
import fire
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models, transforms
from tqdm import tqdm


def train(
    cifar10_dir: str,
    device: str = 'cpu',
    input_size: int = 64,
    batch_size: int = 128,
    num_epochs: int = 300,
    smoothing: float = 0
) -> None:
    """Train a ResNet50 on CIFAR10 (3 cosine cycles) until 90% on val.

    Args:
        cifar10_dir (str): path to torchvision-downloaded Cifar10
        device (str): 'cpu' | 'cuda0' | etc
        input_size (int): number of pixels to resize images to
        batch_size (int): samples per training minibatch
        num_epochs (int): number 0f epochs to train
        smoothing (float): amount of label smoothing
    """
    torch.manual_seed(0)

    # set up training variables
    preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
    ])
    train_loader = DataLoader(
        CIFAR10(
            root=cifar10_dir,
            train=True,
            transform=preproc
        ),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        CIFAR10(
            root=cifar10_dir,
            train=False,
            transform=preproc
        ),
        batch_size=batch_size,
        shuffle=False
    )
    augmenter = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ])
    model = models.resnet50().to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3
    )

    # run main training loop
    prog_bar = tqdm(range(num_epochs))
    val_loss, val_acc = None, None
    for i_epoch in prog_bar:
        for imgs, labels in tqdm(train_loader, leave=False):
            logits = model(augmenter(imgs.to(device)))
            loss = criteria(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prog_bar.set_postfix({
                'loss': float(loss),
                'val-loss': val_loss,
                'vall-acc': val_acc
            })

        # validate every 10 epochs
        if (i_epoch + 1) % 10 == 0:
            model.eval()
            val_logits, val_labels = [], []
            for imgs, labels in tqdm(val_loader, leave=False):
                with torch.no_grad():
                    val_logits.append(model(imgs.to(device)).cpu())
                    val_labels.append(labels.cpu())
            val_logits = torch.cat(val_logits)
            val_pred = val_logits.argmax(dim=1)
            val_labels = torch.cat(val_labels)
            val_loss = float(criteria(val_logits, val_labels))
            val_acc = float((val_pred == val_labels).float().mean())
            del val_logits, val_labels
            model.train()

    # save final weights
    torch.save(model, model.state_dict())


if __name__ == '__main__':
    fire.Fire(train)
