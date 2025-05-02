import torch
from torch.utils.data import DataLoader
from model import ShapeColorClassifier
from dataset import ShapesColorsDataset
import argparse
import pandas as pd
import os
from tqdm import tqdm

# Convert logits to shape-color tuples
def logits_to_pred_tuples(logits, classes, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred = [cls for cls, p in zip(classes, probs) if p > threshold]
    return set(pred)

@torch.no_grad()
def evaluate_and_generate_submission(model, dataloader, classes, output_csv):
    model.eval()
    predictions = []

    for images, img_paths in tqdm(dataloader, desc="Evaluating"):
        images = images.cuda()
        logits = model(images)

        for i in range(images.shape[0]):
            pred_set = logits_to_pred_tuples(logits[i], classes)
            img_path = img_paths[i]  # test image path from dataset

            predictions.append({
                "image_path": img_paths[i],
                "label": repr(sorted(list(pred_set)))
            })                

    # Save CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)

    # Remove trailing newline
    with open(output_csv, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        last_char = f.read(1)
        if last_char == b"\n":
            f.seek(-1, os.SEEK_END)
            f.truncate()    
    print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Shape-color classes in correct order
    classes = [
        ('circle', 'red'), ('circle', 'green'), ('circle', 'blue'),
        ('square', 'red'), ('square', 'green'), ('square', 'blue'),
        ('triangle', 'red'), ('triangle', 'green'), ('triangle', 'blue')
    ]

    # Load model
    model = ShapeColorClassifier(num_classes=len(classes)).cuda()
    model.load_state_dict(torch.load(args.model_ckpt))
    print(f"Loaded model from {args.model_ckpt}")

    # Load dataset
    test_dataset = ShapesColorsDataset(args.test_csv, args.test_dir, mode='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Run evaluation or prediction
    evaluate_and_generate_submission(model, test_loader, classes, args.output_csv)
