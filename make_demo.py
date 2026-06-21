"""Train then render a prediction demo grid on test samples."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from medmnist import NoduleMNIST3D, INFO
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

torch.manual_seed(0); np.random.seed(0)
info = INFO["nodulemnist3d"]
dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("device:", dev, flush=True)

tr = NoduleMNIST3D(split="train", download=True, size=28)
te = NoduleMNIST3D(split="test",  download=True, size=28)

def vol2rgb(v):
    v = v.astype(np.float32)/255.0; c = v.shape[0]//2
    return np.stack([v[c], v[:, c], v[:, :, c]], 0)

class DS(Dataset):
    def __init__(s, raw, train=False):
        s.i = raw.imgs; s.l = raw.labels.astype(np.int64).reshape(-1); s.tr = train
    def __len__(s): return len(s.i)
    def __getitem__(s, k):
        x = torch.from_numpy(vol2rgb(s.i[k])).unsqueeze(0)
        x = F.interpolate(x, 224, mode="bilinear", align_corners=False).squeeze(0)
        if s.tr:
            if torch.rand(1).item() < 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() < 0.5: x = torch.flip(x, [1])
        return (x-0.5)/0.5, s.l[k]

train_loader = DataLoader(DS(tr, True), batch_size=16, shuffle=True)

m = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=2,
    id2label={0: "benign", 1: "malignant"}, label2id={"benign": 0, "malignant": 1})
m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                                 lora_dropout=0.1, bias="none", modules_to_save=["classifier"]))
m.to(dev)
cnt = Counter(DS(tr).l.tolist()); tot = sum(cnt.values())
w = torch.tensor([tot/(2*cnt[c]) for c in [0, 1]], dtype=torch.float32, device=dev)
crit = torch.nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=5e-4, weight_decay=1e-4)

for ep in range(1, 9):
    m.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); crit(m(pixel_values=xb).logits, yb).backward(); opt.step()
    print(f"epoch {ep}/8 done", flush=True)

# ---- prediction demo on chosen test samples ----
lab = te.labels.reshape(-1)
sel = list(np.where(lab == 0)[0][:4]) + list(np.where(lab == 1)[0][:4])  # 4 benign + 4 malignant
m.eval()
fig, axes = plt.subplots(2, 4, figsize=(13, 7))
for ax, i in zip(axes.flat, sel):
    rgb = vol2rgb(te.imgs[i])
    x = torch.from_numpy(rgb).unsqueeze(0)
    x = F.interpolate(x, 224, mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    with torch.no_grad():
        p = torch.softmax(m(pixel_values=x.to(dev)).logits, 1)[0].cpu().numpy()
    pred = int(p.argmax()); true = int(lab[i])
    montage = np.concatenate([rgb[0], rgb[1], rgb[2]], axis=1)  # 3 slices side by side
    ax.imshow(montage, cmap="gray"); ax.axis("off")
    ok = pred == true
    id2 = {0: "benign", 1: "malignant"}
    ax.set_title(f"True: {id2[true]}\nPred: {id2[pred]} ({p[pred]:.2f})  {'OK' if ok else 'X'}",
                 color=("#1f7a1f" if ok else "#b00000"), fontsize=11, fontweight="bold")
plt.suptitle("Prediction demo (ViT+LoRA) — axial | coronal | sagittal", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("fig_demo_predictions.png", dpi=130, bbox_inches="tight")
print("saved fig_demo_predictions.png")
