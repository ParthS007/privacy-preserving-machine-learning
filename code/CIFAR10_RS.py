import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ConvNet
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import sys
sys.path.insert(0, "code/opacus")
from opacus import PrivacyEngine, accountants
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
    ckpt_path = None
    shuffle = True
    pin_memory = True
    verbose = True
    epsilon = 3
    refresh = 1
    final_rate = 0
    batch_partitions = 2
    clip = 1

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Trainer class
class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model = extend(self.model)  # Extend the model
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.criterion = extend(self.criterion)  # Extend the criterion

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        print("Model Saved!")

    def train(self):
        model, config = self.model, self.config
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)

        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                  pin_memory=config.pin_memory, num_workers=config.num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                 num_workers=config.num_workers)

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=config.epsilon,
            max_grad_norm=config.clip,
        )

        num_params = sum(p.numel() for p in model.parameters())
        sampling_rate = config.batch_size / len(train_loader.dataset)
        sigma = accountants.utils.get_noise_multiplier(target_epsilon=config.epsilon, target_delta=1e-5, sample_rate=sampling_rate, epochs=config.max_epochs)

        for epoch in range(config.max_epochs):
            model.train()
            gradient = torch.zeros(size=[num_params]).to(self.device)
            mini_batch = 0

            for iteration, (images, labels) in enumerate(train_loader):
                if iteration % (len(train_loader) // config.refresh) == 0:
                    rate = np.clip(
                        config.final_rate * (
                            epoch * config.refresh + iteration // (len(train_loader) // config.refresh)
                        ) / (config.refresh * config.max_epochs - 1), 0, config.final_rate
                    ) if config.max_epochs > 0 else 0
                    mask = torch.randperm(num_params, device=self.device, dtype=torch.long)[:int(rate * num_params)]

                optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _ = model(images)
                loss = self.criterion(logits, labels)

                batch_grad = []
                with backpack(BatchGrad()):
                    loss.backward()
                
                # Ensure grad_batch is being created
                for p in model.parameters():
                    assert hasattr(p, 'grad_batch'), f"Parameter {p.shape} missing grad_batch"
                    batch_grad.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                    del p.grad_batch

                # clipping gradients
                batch_grad = torch.cat(batch_grad, dim=1)
                for grad in batch_grad:
                    grad[mask] = 0
                norm = torch.norm(batch_grad, dim=1)
                scale = torch.clamp(config.clip / norm, max=1)
                gradient += (batch_grad * scale.view(-1, 1)).sum(dim=0)

                # optimization
                mini_batch += 1
                if mini_batch == config.batch_partitions:
                    gradient = gradient / config.batch_size
                    mini_batch = 0

                    noise = torch.normal(0, sigma * config.clip / config.batch_size, size=gradient.shape).to(self.device)
                    noise[mask] = 0
                    gradient += noise

                    offset = 0
                    for p in model.parameters():
                        shape = p.grad.shape
                        numel = p.grad.numel()
                        p.grad.data = gradient[offset:offset + numel].view(shape)
                        offset += numel

                    optimizer.step()
                    gradient = torch.zeros(size=[num_params]).to(self.device)

                print(f"Epoch {epoch+1}, Iteration {iteration+1}, Loss: {loss.item()}")

            # Save checkpoint after each epoch
            if config.ckpt_path:
                self.save_checkpoint()

model = ConvNet()  # Ensure your ConvNet model is correctly defined
config = TrainingConfig(ckpt_path="./Model.pt", train_type='DPSGD')
trainer = Trainer(model, train_dataset, test_dataset, config)

trainer.train()  # Start training
# trainer.evaluate()

# Optionally save the model
if config.ckpt_path:
    trainer.save_checkpoint()
