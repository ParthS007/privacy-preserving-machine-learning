import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ConvNet
import sys
sys.path.insert(0, "code/opacus")
from opacus import PrivacyEngine
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.5),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

train_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)


class TrainingConfig:
    lr = 3e-4
    betas = (0.9, 0.995)
    weight_decay = 5e-4
    num_workers = 0
    max_epochs = 2
    batch_size = 64
    ckpt_path = None  # Specify a model path here. Ex: "./Model.pt"
    shuffle = True
    pin_memory = True
    verbose = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.train_losses = []
        self.privacy_epsilons=[]
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        print("Model Saved!")

    def train(self):
        model, config = self.model, self.config
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = raw_model.configure_optimizers(config)
        data = self.train_dataset

        train_loader = DataLoader(data, batch_size=config.batch_size,
                            shuffle=config.shuffle,
                            pin_memory=config.pin_memory,
                            num_workers=config.num_workers)

        if config.train_type=='DPSGD':
            delta = 1e-5
            privacy_engine = PrivacyEngine()

            noise_multiplier = 1
            max_grad_norm = 1

            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm, )

            model.train()
            for epoch in range(self.config.max_epochs):
                losses = []
                accuracies = []
                correct = 0
                num_samples = 0
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    num_samples += labels.size(0)
                    logits, loss = model(images, labels)
                    loss = loss.mean()
                    losses.append(loss.item())
                    with torch.no_grad():
                        predictions = torch.argmax(logits,
                                                   dim=1)  # softmax gives prob distribution. Find the index of max prob
                        correct += predictions.eq(labels).sum().item()
                        accuracies.append(correct / num_samples)

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                if config.train_type == 'DPSGD':
                    epsilon = privacy_engine.get_epsilon(delta)
                    self.privacy_epsilons.append(epsilon)
                    print((f"Epoch:{epoch+1} iteration:{epoch+1} | loss:{np.mean(losses)} accuracy:{np.mean(accuracies)} lr:{config.lr} epsilon:{epsilon}"))
                else:
                    print((
                              f"Epoch:{epoch + 1} iteration:{epoch + 1} | loss:{np.mean(losses)} accuracy:{np.mean(accuracies)} lr:{config.lr} "))

                self.train_losses.append(np.mean(losses))
                self.train_accuracies.append(np.mean(accuracies))


            print(self.train_losses)
            print(self.train_accuracies)
            print(self.privacy_epsilons)


model = ConvNet()
config = TrainingConfig(ckpt_path="./Model_NON_DP.pt", train_type='DPSGD')  # example path and training type
trainer = Trainer(model, train_dataset, test_dataset, config)

# Start training
trainer.train()
# trainer.evaluate()

# Optionally save the model
if config.ckpt_path:
    trainer.save_checkpoint()
