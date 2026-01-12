# run_inference_log.py
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image
from train_mnist_cnn import MnistCNN, MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("../../Module 2/data")
OOD_DIR = DATA_DIR / "ood_mnist_digits"
OUT_REF = DATA_DIR / "mnist_reference.parquet"
OUT_PROD = DATA_DIR / "mnist_production.parquet"

SAMPLE_RATE = 0.70  # log ~10% of traffic

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MnistCNN(embedding_dim=ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model

def mnist_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return test_ds

def load_ood_images():
    # Return list of (PIL_image, label_str)
    imgs = []
    for ddir in OOD_DIR.iterdir():
        if not ddir.is_dir():
            continue
        label = ddir.name  # "10", "11", etc.
        for p in ddir.glob("*.png"):
            imgs.append((p, label))
    random.shuffle(imgs)
    return imgs

def img_to_tensor(img: Image.Image):
    # Convert to tensor with same normalization as MNIST
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(img)

def run_and_log(model, mode: str) -> pd.DataFrame:
    """
    mode = "reference" or "production"
    """
    logs: List[Dict[str, Any]] = []
    test_ds = mnist_dataloaders()

    if mode == "reference":
        # Only MNIST digits 0–9
        for idx in range(len(test_ds)):
            x, y_true = test_ds[idx]
            # Simulated online requests: one by one
            if random.random() > SAMPLE_RATE:
                continue  # skip logging this request

            x_batch = x.unsqueeze(0).to(DEVICE)  # (1,1,28,28)
            with torch.no_grad():
                logits, embedding = model(x_batch)
                probs = F.softmax(logits, dim=-1)
                pred_label = int(probs.argmax(dim=-1).item())
                pred_conf = float(probs.max().item())

                logs.append({
                    "request_id": f"ref_{idx}",
                    "mode": "reference",
                    "is_ood": False,
                    "true_label": int(y_true),
                    "pred_label": pred_label,
                    "pred_conf": pred_conf,
                    "probs": probs.cpu().numpy()[0],        # 10-dim vector
                    "embedding": embedding.cpu().numpy()[0],# 64-dim vector
                })

    elif mode == "production":
        # Mix MNIST test with OOD “10–13” images
        ood_imgs = load_ood_images()
        # e.g. 70% MNIST, 30% OOD
        for idx in range(len(test_ds)):
            # First some MNIST 0–9
            x, y_true = test_ds[idx]
            if random.random() < SAMPLE_RATE:
                x_batch = x.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits, embedding = model(x_batch)
                    probs = F.softmax(logits, dim=-1)
                    pred_label = int(probs.argmax(dim=-1).item())
                    pred_conf = float(probs.max().item())

                    logs.append({
                        "request_id": f"prod_mnist_{idx}",
                        "mode": "production",
                        "is_ood": False,
                        "true_label": 'None', # we won't know the true label in production, unless we perform off-line
                        # groundtruth'ing
                        "pred_label": pred_label,
                        "pred_conf": pred_conf,
                        "probs": probs.cpu().numpy()[0],
                        "embedding": embedding.cpu().numpy()[0],
                    })

        for j, (path, label_str) in enumerate(ood_imgs):
            img = Image.open(path).convert("L")
            x = img_to_tensor(img)
            if random.random() > SAMPLE_RATE:
                continue
            x_batch = x.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, embedding = model(x_batch)
                probs = F.softmax(logits, dim=-1)
                pred_label = int(torch.argmax(probs, dim=-1).item())
                pred_conf = float(probs.max().item())
                logs.append({
                    "request_id": f"prod_ood_{j}",
                    "mode": "production",
                    "is_ood": True,
                    "true_label": None,            # unknown in practice
                    "pred_label": pred_label,
                    "pred_conf": pred_conf,
                    "probs": probs.cpu().numpy()[0],
                    "embedding": embedding.cpu().numpy()[0],
                    "ood_label_str": label_str,    # "10","11",...
                })
    else:
        raise ValueError(mode)

    # Convert to DataFrame; we'll explode numpy arrays later if needed
    df = pd.DataFrame(logs)
    return df

def main():
    model = load_model()
    ref_df = run_and_log(model, mode="reference")
    prod_df = run_and_log(model, mode="production")

    # For many monitoring tools it's nicer to split array columns; but some
    # (like Phoenix) can work directly with embedding vectors.
    ref_df.to_parquet(OUT_REF)
    prod_df.to_parquet(OUT_PROD)
    print(f"Saved reference logs to {OUT_REF}")
    print(f"Saved production logs to {OUT_PROD}")

if __name__ == "__main__":
    main()
