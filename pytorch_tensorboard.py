import torch
from torch import nn
from torch import optim
import torch.nn.functional as fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

# region Simple CNN


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fn.relu(self.conv1(x))
        x = self.pool(x)

        x = fn.relu(self.conv2(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

# endregion

# region Hyperparameters


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IN_CHANNELS = 1
NUM_CLASSES = 10
NUM_EPOCHS = 3

# endregion

# region Load Data

train_dataset = datasets.MNIST('data', transform=ToTensor(), download=True)
batch_sizes = [32, 256]
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# endregion


def main() -> None:
    # region Training and TensorBoard

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            step = 0

            model = CNN(in_channels=IN_CHANNELS,
                        num_classes=NUM_CLASSES).to(DEVICE)
            model.train()

            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=0.)
            loss_fn = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            writer = SummaryWriter(
                f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}'
            )

            # Visualize model in TensorBoard
            images: torch.Tensor
            images, _ = next(iter(train_loader))
            writer.add_graph(model, images.to(DEVICE))
            writer.close()

            for epoch in range(1, NUM_EPOCHS + 1):
                losses = []
                accuracies = []

                for batch_idx, batch in enumerate(train_loader):
                    data: torch.Tensor
                    targets: torch.Tensor
                    data, targets = map(lambda x: x.to(DEVICE), batch)

                    # Forward
                    y_pred: torch.Tensor = model(data)
                    loss = loss_fn(y_pred, targets)
                    losses.append(loss.item())

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Calculate Running Training Accuracy
                    features = data.reshape(data.shape[0], -1)
                    img_grid = make_grid(data)
                    _, predictions = y_pred.max(1)

                    num_correct = (predictions == targets).sum()
                    running_training_acc = float(
                        num_correct) / float(data.shape[0])
                    accuracies.append(running_training_acc)

                    # Plot All Things to TensorBoard
                    class_labels = [classes[label] for label in predictions]
                    writer.add_image('mnist_images', img_grid)
                    writer.add_histogram('fc1', model.fc.weight)

                    writer.add_scalar('Training Loss', loss, global_step=step)
                    writer.add_scalar('Training Accuracy',
                                      running_training_acc, global_step=step)

                    if batch_idx == 230:
                        writer.add_embedding(
                            features,
                            metadata=class_labels,
                            label_img=data,
                            global_step=batch_idx
                        )

                    step += 1

                writer.add_hparams(
                    {
                        'lr': learning_rate,
                        'bsize': batch_size
                    },
                    {
                        'accuracy': sum(accuracies) / len(accuracies),
                        'loss': sum(losses) / len(losses)
                    }
                )

    # endregion


if __name__ == '__main__':
    main()
