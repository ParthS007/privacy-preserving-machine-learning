import argparse
import os
import sys
import time
from collections import OrderedDict

import backpack.context
import backpack.core
import numpy as np
import pandas as pd
import torch
import tqdm
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from csv_insights import save_results_to_csv
from losses import CombinedLoss, FocalFrequencyLoss
from matplotlib import pyplot as plt
from networks import get_model
from utils import per_class_dice

from data import get_data

sys.path.insert(0, "code/opacus")

from opacus import PrivacyEngine


def argument_parser():
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument(
        "--num_iterations", default=5, type=int, help="Number of Epochs"
    )
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--n_classes", default=9, type=int)
    parser.add_argument("--ffc_lambda", default=0, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--batch_partitions",
        default=2,
        help="Partition data of a batch to reduce memory footprint.",
        type=int,
    )

    # Opacus specific hyperparameters
    parser.add_argument(
        "--noise_multiplier",
        default=0.5,
        type=float,
        help="Level of independent Gaussian noise into the gradient",
    )
    parser.add_argument(
        "--target_delta", default=1e-5, type=float, help="Target privacy budget δ"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=[1.0] * 64,  # 1.0 for flat clipping mode
        type=float,
        help="Per-sample gradient clipping threshold",
    )
    parser.add_argument(
        "--clipping_mode",
        default="per_layer",
        choices=["flat", "per_layer", "adaptive"],
        help="Gradient clipping mode",
    )
    # Random Sparsification
    parser.add_argument(
        "--final_rate",
        default=0,
        type=float,
        help="Sparsification rate at the end of gradual cooling.",
    )
    parser.add_argument(
        "--refresh",
        default=1,
        type=int,
        help="Randomization times of sparsification mask per epoch.",
    )

    # Dataset options
    parser.add_argument("--dataset", default="Duke", choices=["Duke"])
    parser.add_argument("--image_size", default="224", type=int)
    parser.add_argument(
        "--image_dir",
        default="data/DukeData/",
        choices=["data/DukeData"],
    )
    parser.add_argument("--model_name", default="unet", choices=["unet", "NestedUNet"])

    # Network options
    parser.add_argument("--g_ratio", default=0.5, type=float)

    # Other options
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--in_channels", default=1, type=int)

    return parser


def colored_text(st):
    return "\033[91m" + st + "\033[0m"


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval(
    val_loader,
    criterion,
    model,
    n_classes,
    dataset,
    algorithm,
    location,
    dice_s=True,
    device="cpu",
    im_save=False,
):
    """
    The dice score is a measure of the overlap between the predicted segmentation and the ground truth segmentation.
    A higher dice score indicates a better overlap.
    The dice score is calculated for each class separately, and the mean dice score is also calculated.
    The loss is calculated using the specified loss function as parameter `criterion`.
    """
    model.eval()
    loss = 0
    counter = 0
    dice = 0

    dice_all = np.zeros(n_classes)

    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

        pred = model(img)
        max_val, idx = torch.max(pred, 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

        if dice_s:
            d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
            dice += d1
            dice_all += d2

        loss += criterion(pred, label.squeeze(1), device=device).item()

        if im_save:
            # Save the predicted segmentation and the ground truth segmentation
            name = f"predicted_segment_{counter}_for_{dataset}_with_{algorithm}.png"
            fig, ax = plt.subplots(1, 2)
            fig.suptitle(name, fontsize=10)

            ax[0].imshow(label.cpu().squeeze().numpy(), cmap="gray")
            ax[0].set_title(f"Ground Truth")
            ax[1].imshow(idx.cpu().squeeze().numpy(), cmap="gray")
            ax[1].set_title(f"Prediction")
            fig.subplots_adjust(top=0.85)

            dir_path = f"results/{algorithm}/{location}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.savefig(f"results/{algorithm}/{location}/{name}.png")
            plt.close(fig)

        counter += 1

    loss = loss / counter
    dice = dice / counter
    dice_all = dice_all / counter
    print("Validation loss: ", loss, " Mean Dice: ", dice.item(), "Dice All:", dice_all)
    return dice, loss


def train(args):
    # Optimization hyperparameters
    batch_size = args.batch_size
    iterations = args.num_iterations
    learning_rate = args.learning_rate
    n_classes = args.n_classes

    # Opacus specific hyperparameters
    noise_multiplier = args.noise_multiplier
    target_delta = args.target_delta
    max_grad_norm = args.max_grad_norm
    clipping_mode = args.clipping_mode
    algorithm = "opacus-rs-ac"

    # Random Sparsification specific hyperparameters
    final_rate = args.final_rate
    refresh = args.refresh

    # Dataset options
    dataset = args.dataset
    img_size = args.image_size
    data_path = args.image_dir
    model_name = args.model_name

    # Network options
    ratio = args.g_ratio

    # Other options
    device = args.device

    training_losses = []
    validation_losses = []
    validation_dice_scores = []
    criterion_seg = CombinedLoss()
    criterion_seg = extend(criterion_seg, debug=True)
    criterion_ffc = FocalFrequencyLoss()
    save_name = f"results/{algorithm}/{model_name}_{noise_multiplier}.pt"
    file_name = f"results/{algorithm}/{model_name}_{dataset}.csv"
    location = f"{noise_multiplier}"

    max_dice = 0
    best_test_dice = 0
    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    model = extend(model, debug=True)
    model.train()
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = (
        get_data(data_path, img_size, batch_size)
    )
    optimizer = torch.optim.SGD(
        list(model.parameters()), lr=learning_rate, weight_decay=args.weight_decay
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, data_path = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        clipping=clipping_mode,
    )

    num_params = sum(p.numel() for p in model.parameters())
    start_time = time.time()
    print(f"\t Training start time: {start_time}")
    iteration_train_times = []
    overall_privacy_spent = []
    train_epoch_losses = []
    mini_batch = 0
    for t in range(iterations):
        start_epoch_time = time.time()
        print(f"\t Training start epoch time: {start_epoch_time}")

        total_loss = 0
        total_samples = 0

        # Random Sparsification
        gradient = torch.zeros(size=[num_params]).to(device)

        for img, label in tqdm.tqdm(train_loader):
            # Compute current gradual exit rate
            if iterations % (len(train_loader) // args.refresh) == 0:
                rate = (
                    np.clip(
                        args.final_rate
                        * (
                            t * args.refresh
                            + iterations // (len(train_loader) // args.refresh)
                        )
                        / (args.refresh * args.epochs - 1),
                        0,
                        args.final_rate,
                    )
                    if args.epochs >= 0
                    else 0
                )
                mask = torch.randperm(num_params, device=device, dtype=torch.long)[
                    : int(rate * num_params)
                ]

            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(
                label, num_classes=n_classes
            ).squeeze()

            # Training
            optimizer.zero_grad()

            # Computing gradients
            pred = model(img)
            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            loss = criterion_seg(
                pred, label.squeeze(1), device=device
            ) + args.ffc_lambda * criterion_ffc(pred_oh, label_oh)

            batch_grad = []
            with backpack(BatchGrad()):
                loss.backward()
            for p in model.parameters():
                batch_grad.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                del p.grad_batch

            # Clipping gradients
            batch_grad = torch.cat(batch_grad, dim=1)
            for grad in batch_grad:
                grad[mask] = 0
            norm = torch.norm(batch_grad, dim=1)
            scale = torch.clamp(max_grad_norm / norm, max=1)
            gradient += (batch_grad * scale.view(-1, 1)).sum(dim=0)

            # Optimization
            mini_batch += 1
            if mini_batch == args.batch_partitions:
                gradient = gradient / args.batch_size
                mini_batch = 0

                # Perturbation
                noise = torch.normal(
                    0,
                    noise_multiplier * max_grad_norm / args.batch_size,
                    size=gradient.shape,
                ).to(device)
                noise[mask] = 0
                gradient += noise

                # Replace the gradients with the perturbed gradients
                offset = 0
                for p in model.parameters():
                    shape = p.grad.shape
                    numel = p.grad.numel()
                    p.grad.data = gradient[offset : offset + numel].view(shape)
                    offset += numel

                optimizer.step()
                gradient = torch.zeros(size=[num_params]).to(device)

            total_loss = loss.item() + img.size(0)
            total_samples += img.size(0)

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        iteration_train_times.append(epoch_time)
        average_loss = total_loss / total_samples
        training_losses.append(average_loss)
        train_epoch_losses.append(loss.item())
        print(
            f"\tTrain Epoch: [{t + 1}/{iterations}] \t"
            f"Train Average Loss: {np.mean(average_loss):.6f} \t"
            f"Train Epoch Time: {epoch_time:.2f} \t"
            f"Train Epoch loss: {loss.item():.6f} \t"
        )

        # Calculate privacy spent
        privacy_spent = privacy_engine.get_epsilon(delta=target_delta)
        overall_privacy_spent.append(privacy_spent)

        if t % 2 == 0:
            print(loss.item())

        if t % 10 == 0 or t > 4:
            print("Epoch", t, "/", iterations)
            print("Validation in progress...")
            dice, validation_loss = eval(
                val_loader,
                criterion_seg,
                model,
                dice_s=True,
                n_classes=n_classes,
                dataset=dataset,
                algorithm=algorithm,
                location=f"{noise_multiplier}",
                im_save=True,
            )
            validation_losses.append(validation_loss)
            validation_dice_scores.append(dice)

            if dice > max_dice:
                max_dice = dice
                best_iter = t
                print(colored_text("Updating model, epoch: "), t)
                print(f"Best iteration: {best_iter}, Best Dice: {max_dice}")
                torch.save(model.state_dict(), save_name)

            model.train()

    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: ", training_time)

    training_losses_str = str(training_losses)
    validation_losses_str = str(validation_losses)
    validation_dice_scores_str = str(validation_dice_scores)
    print("Training Losses: ", training_losses_str)
    print("Validation Losses: ", validation_losses_str)
    print("Validation Dice Scores: ", validation_dice_scores_str)
    print("Overall Privacy Spent: ", overall_privacy_spent)

    # Plotting Privacy Epsilon Over Time
    epochs = list(range(1, iterations + 1))
    name = f"Privacy Epsilon Over Time for '{dataset}' dataset with {algorithm} ."
    plt.plot(epochs, overall_privacy_spent)
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.title(name)
    plt.savefig(f"results/{algorithm}/{location}/{name}.png")
    plt.show()

    # Privacy Epsilon vs Training Loss
    name = (
        f"Privacy Epsilon vs. Training Loss for '{dataset}' dataset with {algorithm} ."
    )
    plt.plot(overall_privacy_spent, training_losses)
    plt.xlabel("Epsilon")
    plt.ylabel("Training Loss")
    plt.title(name)
    plt.savefig(f"results/{algorithm}/{location}/{name}.png")
    plt.show()

    # Training Average Loss Over Time
    name = f"Training Average Loss Over Time for '{dataset}' dataset with {algorithm} ."
    plt.figure()
    plt.plot(epochs, training_losses, label="Train Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(name)
    plt.legend()
    plt.savefig(f"results/{algorithm}/{location}/{name}.png")
    plt.show()

    # Training Epoch Time Over Time
    name = f"Training Epoch Time Over Time for '{dataset}' dataset with {algorithm} ."
    plt.figure()
    plt.plot(epochs, iteration_train_times, label="Train Epoch Time")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(name)
    plt.legend()
    plt.savefig(f"results/{algorithm}/{location}/{name}.png")
    plt.show()

    # Training Epoch Loss Over Time
    name = f"Training Epoch Loss Over Time for '{dataset}' dataset with {algorithm} ."
    plt.figure()
    plt.plot(epochs, train_epoch_losses, label="Train Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(name)
    plt.legend()
    plt.savefig(f"results/{algorithm}/{location}/{name}.png")
    plt.show()

    # Iteration Time Per Epoch Over Time
    name = (
        f"Iteration Time Per Epoch Over Time for '{dataset}' dataset with {algorithm} ."
    )
    plt.figure()
    plt.plot(
        range(1, len(iteration_train_times) + 1),
        iteration_train_times,
        label="Iteration Train Time",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title(name)
    plt.legend()
    plt.savefig(f"results/{algorithm}/{location}/{name}.png")
    plt.show()

    save_results_to_csv(
        # Location
        file_name=file_name,
        # Optimization hyperparameters
        batch_size=batch_size,
        epochs=iterations,
        learning_rate=learning_rate,
        # Opacus specific hyperparameters
        noise_multiplier=noise_multiplier,
        # target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        clipping_mode=clipping_mode,
        algorithm=algorithm,
        overall_privacy_spent=overall_privacy_spent,
        # Dataset options
        dataset=dataset,
        model_name=model_name,
        # Other options
        device=device,
        # Results Metrics
        iteration_train_times=iteration_train_times,
        training_losses=training_losses,
        validation_losses=validation_losses,
        validation_dice_scores=validation_dice_scores,
        total_training_time=training_time,
    )
    return model


if __name__ == "__main__":
    args = argument_parser().parse_args()
    print(args)
    set_seed(args.seed)

    train(args)
