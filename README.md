# 🟥🟩🔵 Shapes and Colors Prediction Challenge

This repository contains my solution for the **Shapes and Colors Prediction** take-home assignment. The task is to build a multi-label image classifier that detects geometric shapes (circle, square, triangle) and their colors (red, green, blue) from synthetic images.

Please see `./Shapes_colors_Report.ipynb` for complete description of the task.

---

## Task Description

Each input image contains a set of non-overlapping geometric shapes — circles, squares, and triangles — in red, green, or blue. The model predicts a set of `(shape, color)` tuples per image.

Example:

```
Input: test_dataset/img_1.png  
Prediction: [('circle', 'red'), ('square', 'blue')]
```

---

## Dataset

- **Images**: Synthetic RGB images of size 128×128, with random rotation, color, and placement.
- **Labels**: A list of `(shape, color)` tuples.
- **Possible Classes (9 total)**:  
  - Shapes: `circle`, `square`, `triangle`  
  - Colors: `red`, `green`, `blue`  
  → Combined as: `[('circle', 'red'), ..., ('triangle', 'blue')]`

---

## Model

A modified **ResNet-18** is used for multi-label classification:
- Final fully connected layer is replaced with a 9-unit output layer.
- Each output corresponds to one of the 9 shape–color combinations.
- Training uses `BCEWithLogitsLoss` with sigmoid activation.

---

## Training Pipeline

```bash
python train.py \
  --train_csv train_v2.csv \
  --train_dir dataset_v2 \
  --num_epochs 100 \
  --batch_size 128 \
  --val_step 32 \
  --max_lr 6e-4 \
  --min_lr 5e-5 \
  --final_lr 1e-6 \
  --warmup_steps 2000 \
  --final_linear_decay \
  --out_path ./output_run1
```

---

## Evaluation Metric

The primary metric is **Jaccard Similarity** between predicted and true sets:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

- `J = 1.0`: perfect prediction  
- `J = 0.0`: no overlap

This metric suits multi-label classification and reflects how well predicted shape-color pairs match the ground truth.

---

## Submission Format

Output predictions are stored in `./output_bz128_warm1500_nepoch500_maxlr8e-4_minlr5e-5_wdcay_3e-4/submission.csv` as:

```csv
image_path,label
test_dataset_v2/img_0.png,"[('square', 'blue'), ('triangle', 'blue')]"
test_dataset_v2/img_1.png,"[('circle', 'green')]"
```

To submit:

```bash
kaggle competitions submit -c all-shapes-and-colors -f submission.csv -m "Final submission"
```

---

## Findings

- **Validation loss spikes** occur due to rare samples  
- After step ~12,000, training stabilizes, and validation loss reaches a **flat minimum**
- Training and validation loss **track closely** — no sign of overfitting
- The final model achieves high and consistent **Jaccard Similarity**

---

## Report

The accompanying Jupyter notebook `Shapes_Colors_Report.ipynb` includes:
- Architecture overview
- Loss & metric plots
- Evaluation with failure case analysis

---

## Structure

```
.
├── dataset.py              # Custom dataset class
├── model.py                # ResNet-based classifier
├── train.py                # Training loop
├── eval.py                 # Prediction script
├── scheduler.py            # Customized warmup + cosine annealing + opational linear decay
├── submission.csv          # Output predictions
├── Shapes_Colors_Report.ipynb # Final Report
├── figure                  # For figures storage
├── output_bz128_warm1500_nepoch500_maxlr8e-4_minlr5e-5_wdcay_3e-4 # Storing the best result.
├── output_test_no_skip_empty # For screening the cases without empty skip.
└── README.md               # This file
```

---

## Author

**Patrick Chen**  
Carnegie Mellon University  
Contact: bochunc@andrew.cmu.edu

---

## Status

- Completed and submitted  
- Further improvements possible with data augmentation or deeper models