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
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import models, transforms
from tqdm import tqdm


def train(
    cifar10_dir: str,
    device: str = 'cpu',
    input_size: int = 64,
    batch_size: int = 128,
    num_epochs: int = 225,
    smoothing: float = 0
) -> None:
    """Train a ResNet50 on CIFAR10 (3 cosine cycles) until 90% on val.

    Args:
        cifar10_dir (str): path to torchvision-downloaded Cifar10
        device (str): 'cpu' | 'cuda:0' | etc
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
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        num_epochs // 3
    )

    # set up tensorboard
    step = 0
    board = SummaryWriter(log_dir='tensorboard/train')
    val_board = SummaryWriter(log_dir='tensorboard/val')

    # run main training loop
    prog_bar = tqdm(range(num_epochs))
    val_loss, val_acc = None, None
    for i_epoch in prog_bar:
        # train
        for imgs, labels in tqdm(train_loader, leave=False):
            logits = model(augmenter(imgs.to(device)))
            loss = criteria(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prog_bar.set_postfix({
                'loss': float(loss),
                'val-loss': val_loss,
                'val-acc': val_acc
            })

            step += 1
            board.add_scalar('loss', loss, step)
        scheduler.step()

        # validate
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
        val_board.add_scalar('loss', val_loss, step)
        val_board.add_scalar('acc', val_acc, step)
        del val_logits, val_labels
        model.train()
        if val_acc > 0.8:
            torch.save(model.state_dict(), f'ls{smoothing}-epoch{i_epoch}.pt')

    # save final weights
    torch.save(model.state_dict(), f'ls-{smoothing}.pt')


if __name__ == '__main__':
    fire.Fire(train)
