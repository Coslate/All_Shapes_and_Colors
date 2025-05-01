import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import numpy as np
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import ShapesColorsDataset
from model import ShapeColorClassifier
from scheduler import CustomScheduler

def jaccard_index(pred_set, true_set):
    intersection = len(pred_set & true_set)
    union = len(pred_set | true_set)
    return intersection / union if union > 0 else 1.0

def save_checkpoint(path, model, optimizer, scheduler, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
        'epoch': epoch
    }, path)
    print(f"[*] Model Checkpoint saved to {path}")    

def load_checkpoint(path, model, optimizer, scheduler, args):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load the .npz file
    loaded_data = np.load(os.path.join(args.out_path, f"{os.path.basename(args.load_metric_file)}"))
    train_losses = list(loaded_data["train_losses"])
    train_steps = list(loaded_data["train_steps"])
    val_losses = list(loaded_data["val_losses"])
    val_steps = list(loaded_data["val_steps"])
    val_metric = list(loaded_data["val_metric"])
    val_metric_steps = list(loaded_data["val_metric_steps"])

    print(f"[+] Loaded MLP/Unet checkpoint from {path}")
    return checkpoint.get('epoch', 0), train_losses, train_steps, val_losses, val_steps, val_metric, val_metric_steps 


def train_model(args):
    # Dataset & Dataloader
    os.makedirs(args.out_path, exist_ok=True)    
    dataset = ShapesColorsDataset(args.train_csv, args.train_dir)
    print(f"len(dataset) = {len(dataset)}")
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                  generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Model & Optimizer & Scheduler
    model = ShapeColorClassifier().cuda()
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    total_steps = args.num_epochs * len(train_loader)
    scheduler = CustomScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,#1e-4,
        max_lr=args.max_lr,
        final_lr=args.final_lr,#1e-6,
        T_0=total_steps,
        T_mult=1,
        final_linear_decay=args.final_linear_decay
    )
    loss_fn = BCEWithLogitsLoss()
    total_steps = args.num_epochs * len(train_loader)

    # Track losses
    start_epoch = 0
    global_step = 0
    train_losses = []
    val_losses = []
    train_steps = []
    val_steps = []
    val_metric = []
    val_metric_steps = []
    lr_values = []
    lr_steps = []
    grad_norms = []
    grad_steps = []

    classes = [
        ('circle', 'red'), ('circle', 'green'), ('circle', 'blue'),
        ('square', 'red'), ('square', 'green'), ('square', 'blue'),
        ('triangle', 'red'), ('triangle', 'green'), ('triangle', 'blue')
    ]    

    # Load model checkpoint if provided
    if args.resume_model and os.path.exists(args.load_model_ckpt):
        start_epoch, train_losses, train_steps, val_losses, val_steps, val_metric, val_metric_steps = load_checkpoint(args.load_model_ckpt, model, optimizer, scheduler)
        global_step = start_epoch * len(train_loader)

    model.train()
    best_jaccard = -1
    for epoch in range(args.num_epochs):
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.cuda(), labels.cuda()
            non_empty_mask = labels.sum(dim=1) > 0
            if non_empty_mask.sum() == 0:
                continue  # skip this batch entirely

            # Filter images and labels
            images = images[non_empty_mask]
            labels = labels[non_empty_mask]

            logits = model(images)
            logits = torch.clamp(logits, min=-10, max=10)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_threshold)
            optimizer.step()
            scheduler.step(global_step)
            total_loss += loss.item()
            global_step += 1
            train_loss_vis = total_loss / (global_step+1)
            pbar.set_description(
                "Loss {:.5f} | Avg. Loss {:.5f}".format(loss.item(), train_loss_vis)
            )

            # Track LR
            current_lr = scheduler.get_last_lr()[0]
            lr_steps.append(global_step)
            lr_values.append(current_lr)

            # Track Grad Norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_steps.append(global_step)
            grad_norms.append(total_norm)

            # show on tqdm
            #pbar.set_postfix(loss=loss.item(), lr=current_lr, grad_norm=total_norm)            

            # Validation
            if global_step % args.val_step == 0:
                model.eval()
                val_loss = 0
                jaccard_scores = []
                with torch.no_grad():
                    val_batches_used = 0
                    for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
                        images, labels = images.cuda(), labels.cuda()
                        logits = model(images)

                        non_empty_mask = labels.sum(dim=1) > 0
                        if non_empty_mask.sum() == 0:
                            continue  # skip batch if all are empty

                        # Filter inputs to non-empty samples
                        val_batches_used += 1
                        logits_filtered = logits[non_empty_mask]
                        labels_filtered = labels[non_empty_mask]
                        probs_filtered = torch.sigmoid(logits_filtered)
                        loss = loss_fn(logits_filtered, labels_filtered)
                        val_loss += loss.item()

                        # Compute Jaccard similarity
                        for i in range(probs_filtered.shape[0]):
                            pred = set([cls for cls, p in zip(classes, probs_filtered[i]) if p > 0.5])
                            true = set([cls for cls, v in zip(classes, labels_filtered[i]) if v == 1])
                            jaccard_scores.append(jaccard_index(pred, true))

                        if loss.item() > args.threshold:  # e.g., 10.0
                            print(f"Unstable prediction at step {global_step}, loss = {loss.item():.2f}")
                            non_empty_indices = torch.where(non_empty_mask)[0]  # e.g., tensor([0, 3, 5])
                            # Identify high-loss samples in batch
                            for i, real_idx in enumerate(non_empty_indices):
                                sample_loss = loss_fn(logits_filtered[i].unsqueeze(0), labels_filtered[i].unsqueeze(0)).item()
                                if sample_loss > args.threshold:
                                    dataset_index = batch_idx * val_loader.batch_size + real_idx.item()
                                    print(f"[!] High loss: {sample_loss:.2f} at validation index {dataset_index}")
            
                                    # Extract raw data (on CPU for saving or debugging)
                                    img_cpu = images[real_idx].cpu()
                                    label_cpu = labels[real_idx].cpu()
                                    logit_cpu = logits[real_idx].cpu()
            
                                    # Optional: visualize or save
                                    save_path = f"{args.out_path}/high_loss_{dataset_index}.png"
                                    torchvision.utils.save_image(img_cpu, save_path)
                                    print(f"Saved image to {save_path}")
                                    print(f"Label: {label_cpu.tolist()}")
                                    print(f"Logits: {logit_cpu.tolist()}")

                    val_avg = val_loss / val_batches_used
                    val_metric_value = np.mean(jaccard_scores)
                    val_losses.append(val_avg)
                    val_steps.append(global_step)
                    val_metric.append(val_metric_value)
                    val_metric_steps.append(global_step)
                    if val_metric_value > best_jaccard:
                        best_jaccard = val_metric_value
                        out_file = os.path.join(args.out_path, f"{global_step}_shape_color_model.pth")
                        torch.save(model.state_dict(), f"{out_file}")
                        print(f"best jaccard = {val_metric_value}, saving: {out_file}")

                    np.savez((os.path.join(args.out_path, f"{os.path.basename(args.load_metric_file)}")),
                    train_losses=train_losses,
                    val_losses=val_losses,
                    val_steps=val_steps,
                    val_metric_losses=val_metric,
                    val_metric_steps=val_metric_steps
                    )
                    print(f"[Epoch {epoch+1}] Train Loss: {train_loss_vis:.5f} | Val Loss: {val_avg:.5f} | Jaccard: {val_metric_value:.5f}")

        train_avg = total_loss / len(train_loader)
        train_losses.append(train_avg)
        train_steps.append(global_step)
        model.train()

    # Save model
    torch.save(model.state_dict(), os.path.join(args.out_path, f"final_shape_color_model.pth"))

    # === Save Train Loss & Validation Loss Plot ===
    global_steps = np.arange(1, args.num_epochs * len(train_loader) + 1)

    # Interpolate all losses and metrics
    train_interp = np.interp(global_steps, train_steps, train_losses)
    val_interp = np.interp(global_steps, val_steps, val_losses)
    jaccard_interp = np.interp(global_steps, val_metric_steps, val_metric)

    # Plott Train/Val Loss vs Steps
    plt.figure(figsize=(10, 6))

    # Training Loss
    plt.plot(train_steps, train_losses, label="Training Loss (Actual)", color="blue", linestyle='-', marker='o')
    plt.plot(global_steps, train_interp, label="Train Loss (Interpolated)", color="blue", linestyle='--', alpha=0.5)

    # Validation Loss
    plt.plot(val_steps, val_losses, label="Validation Loss (Actual)", color="crimson", linestyle='-', marker='s')
    plt.plot(global_steps, val_interp, label="Validation Loss (Interpolated)", color="crimson", linestyle='--', alpha=0.5)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss and Jaccard Metric vs Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_path, "training_validation_loss_vs_steps.png"))
    plt.show()
    plt.close()

    # Plot Jaccard Metric vs Steps
    plt.figure(figsize=(10, 6))
    plt.plot(val_metric_steps, val_metric, label="Jaccard Score (Actual)", color="darkgreen", linestyle='-', marker='x')
    plt.plot(global_steps, jaccard_interp, label="Jaccard Score (Interpolated)", color="darkgreen", linestyle='--', alpha=0.5)
    plt.xlabel("Steps")
    plt.ylabel("Score")
    plt.title("Jaccard Metric vs Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_path, "metric_jaccard_vs_steps.png"))
    plt.show()
    plt.close()

    # Plot Learning Rate vs Steps
    plt.figure(figsize=(10, 6))
    plt.plot(lr_steps, lr_values, label="Learning Rate", color="purple")
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_path, "learning_rate_vs_steps.png"))
    plt.show()
    plt.close()

    # Plot Gradient Norm vs Steps
    plt.figure(figsize=(10, 6))
    plt.plot(grad_steps, grad_norms, label="Gradient Norm (L2)", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm vs Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_path, "grad_norm_vs_steps.png"))
    plt.show()
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--final_linear_decay", action="store_true", help="Whether to do linear decay after cosin annealing.")
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_path", type=str, default="./output", help="Directory to save results")
    parser.add_argument("--val_step", type=int, default=100, help="Validation frequency (in steps)")
    parser.add_argument("--resume_model", action="store_true", help="Whether to resume training from model checkpoint")
    parser.add_argument("--load_model_ckpt", type=str, default="./loaded_model.pth", help="Path to model checkpoint")
    parser.add_argument('--load_metric_file', default='metrics_loss_data.npz', type=str)
    parser.add_argument("--warmup_steps", type=int, default=1500, help="Number of warmup steps for LR scheduler") 
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--grad_norm_threshold", type=float, default=1.0)
#3%*num_epochs*len(dataloader)
    args = parser.parse_args()

    train_model(args)