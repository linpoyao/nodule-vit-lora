"""Generate the NoduleMNIST3D + ViT(LoRA) Jupyter notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------------------
md(r"""# 肺結節良/惡性分類 — NoduleMNIST3D × Vision Transformer (LoRA 微調)

本 notebook 完整示範一個醫學影像深度學習流程:

1. **主題**:醫學影像辨識(肺部 CT 結節)
2. **公開資料庫**:[MedMNIST v2](https://medmnist.com/) 的 **NoduleMNIST3D**(僅約 1.6k 筆,屬於小型資料庫,適合作業/實驗)
3. **目標**:肺結節 **良性 vs 惡性** 二分類
4. **模型**:**Vision Transformer (ViT-base)**,使用 ImageNet-21k 預訓練權重
5. **Clone model**:透過 HuggingFace `transformers` 下載預訓練 ViT
6. **Fine-tune (PEFT: LoRA)**:用 `peft` 只訓練少量 LoRA 參數 + 分類頭
7. **任務類型**:**Classification**

> **3D → ViT 的關鍵設計**:ViT 是 2D 模型,而 NoduleMNIST3D 是 28×28×28 的 3D 體積。
> 我們取每個體積的 **三個正交中心切片**(axial / coronal / sagittal),
> 堆疊成 RGB 三通道影像,再 resize 到 224×224 餵給預訓練 ViT。
> 這樣既能用上強大的預訓練權重,又保留了 3D 的多視角資訊。
""")

# ---------------------------------------------------------------------------
md("""## 0. 環境檢查與安裝

若在 Colab 執行,先取消下方安裝指令的註解。本機若已安裝可略過。""")

code("""# 如在 Colab 請取消註解執行:
# !pip install -q medmnist transformers peft scikit-learn matplotlib

import torch, transformers, peft, medmnist, sklearn
print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("peft         :", peft.__version__)
print("medmnist     :", medmnist.__version__)
print("sklearn      :", sklearn.__version__)""")

code("""# 選擇運算裝置:優先 CUDA(Colab GPU) > MPS(Mac GPU) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("使用裝置:", device)""")

# ---------------------------------------------------------------------------
md("""## 1. 載入 NoduleMNIST3D

`medmnist` 會自動下載資料(約幾 MB)。我們直接取用原始 numpy 陣列
`.imgs` (N, 28, 28, 28) 與 `.labels` (N, 1),自行建立 Dataset,
避免不同版本 `__getitem__` 行為差異。""")

code("""from medmnist import NoduleMNIST3D
from medmnist import INFO

info = INFO["nodulemnist3d"]
print("任務類型 :", info["task"])
print("類別     :", info["label"])
print("通道數   :", info["n_channels"])

# size=28 為 MNIST-like 最小版本(資料量小,適合 CPU/作業)
train_raw = NoduleMNIST3D(split="train", download=True, size=28)
val_raw   = NoduleMNIST3D(split="val",   download=True, size=28)
test_raw  = NoduleMNIST3D(split="test",  download=True, size=28)

print("\\ntrain imgs:", train_raw.imgs.shape, "labels:", train_raw.labels.shape)
print("val   imgs:", val_raw.imgs.shape)
print("test  imgs:", test_raw.imgs.shape)
print("影像 dtype:", train_raw.imgs.dtype, "值域:", train_raw.imgs.min(), "~", train_raw.imgs.max())""")

code("""import numpy as np
# 類別分布(檢查是否不平衡)
for name, ds in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
    vals, cnts = np.unique(ds.labels, return_counts=True)
    print(f"{name:5s} -> " + ", ".join(f"{info['label'][str(v)]}={c}" for v, c in zip(vals, cnts)))""")

# ---------------------------------------------------------------------------
md("""## 2. 3D → 2D 前處理:三正交中心切片當 RGB

對每個 28×28×28 體積:
- **axial**:沿第 0 軸(深度)中央切片
- **coronal**:沿第 1 軸中央切片
- **sagittal**:沿第 2 軸中央切片

三張灰階切片堆成 (3, 28, 28) 的 RGB,resize 到 224×224,
並用 ViT 的 mean/std=0.5 正規化。訓練時加入隨機翻轉做簡單擴增。""")

code("""import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 224
MEAN = 0.5   # google/vit-base-patch16-224-in21k 的正規化參數
STD  = 0.5

def volume_to_rgb(vol):
    \"\"\"(28,28,28) uint8 -> (3,28,28) float32 取三正交中心切片\"\"\"
    vol = vol.astype(np.float32) / 255.0
    c = vol.shape[0] // 2
    axial    = vol[c, :, :]
    coronal  = vol[:, c, :]
    sagittal = vol[:, :, c]
    return np.stack([axial, coronal, sagittal], axis=0)  # (3,28,28)

class NoduleSliceDataset(Dataset):
    def __init__(self, raw, train=False):
        self.imgs = raw.imgs
        self.labels = raw.labels.astype(np.int64).reshape(-1)
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rgb = volume_to_rgb(self.imgs[idx])           # (3,28,28)
        x = torch.from_numpy(rgb).unsqueeze(0)        # (1,3,28,28)
        x = F.interpolate(x, size=IMG_SIZE, mode="bilinear", align_corners=False)
        x = x.squeeze(0)                              # (3,224,224)
        if self.train:                                 # 簡單資料擴增
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
        x = (x - MEAN) / STD
        return x, self.labels[idx]

train_ds = NoduleSliceDataset(train_raw, train=True)
val_ds   = NoduleSliceDataset(val_raw)
test_ds  = NoduleSliceDataset(test_raw)

BATCH = 16
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH)
test_loader  = DataLoader(test_ds,  batch_size=BATCH)

xb, yb = next(iter(train_loader))
print("一個 batch:", xb.shape, yb.shape, "| x 值域:", round(xb.min().item(),2), "~", round(xb.max().item(),2))""")

# ---------------------------------------------------------------------------
md("""## 3. 視覺化幾個樣本

左到右為 axial / coronal / sagittal 切片;標題顯示良惡性標籤。""")

code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    rgb = volume_to_rgb(train_raw.imgs[i])      # (3,28,28)
    montage = np.concatenate([rgb[0], rgb[1], rgb[2]], axis=1)  # 並排三切片
    ax.imshow(montage, cmap="gray")
    lbl = info["label"][str(int(train_raw.labels[i][0]))]
    ax.set_title(lbl, fontsize=10)
    ax.axis("off")
plt.suptitle("NoduleMNIST3D:axial | coronal | sagittal 中心切片")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------------------
md("""## 4. Clone 預訓練 ViT 模型

從 HuggingFace 下載 `google/vit-base-patch16-224-in21k`(ImageNet-21k 預訓練)。
換上 2 類的新分類頭(隨機初始化,稍後會訓練)。""")

code("""from transformers import ViTForImageClassification

CKPT = "google/vit-base-patch16-224-in21k"
id2label = {0: info["label"]["0"], 1: info["label"]["1"]}
label2id = {v: k for k, v in id2label.items()}

base_model = ViTForImageClassification.from_pretrained(
    CKPT,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
n_total = sum(p.numel() for p in base_model.parameters())
print(f"ViT 全模型參數量:{n_total/1e6:.1f} M")
print("分類頭:", base_model.classifier)""")

# ---------------------------------------------------------------------------
md("""## 5. 套用 LoRA (PEFT)

只在 attention 的 `query`、`value` 注入低秩 (rank=8) adapter,
凍結原始權重;分類頭用 `modules_to_save` 設為完整可訓練
(因為它是新初始化的)。可看到可訓練參數只佔極小比例。""")

code("""from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to(device);""")

# ---------------------------------------------------------------------------
md("""## 6. 訓練

使用 AdamW + CrossEntropy。資料若類別不平衡,加入 class weight。
CPU/MPS 上 epoch 數設小;Colab GPU 可調大。""")

code("""from collections import Counter

# 類別權重處理不平衡
cnt = Counter(train_ds.labels.tolist())
total = sum(cnt.values())
weights = torch.tensor([total/(2*cnt[c]) for c in [0,1]], dtype=torch.float32, device=device)
print("class weights:", weights.tolist())

criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=5e-4, weight_decay=1e-4)

EPOCHS = 8  # Colab GPU 可調到 15-20

@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(pixel_values=xb).logits
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
    return correct / total

for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(pixel_values=xb).logits
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_ds)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} | val_acc {val_acc:.4f}")""")

# ---------------------------------------------------------------------------
md("""## 7. 測試集評估

報告 Accuracy、AUC、混淆矩陣與 classification report。
醫學影像不平衡時 **AUC** 比 accuracy 更具參考價值。""")

code("""from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

model.eval()
all_logits, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(pixel_values=xb).logits.cpu()
        all_logits.append(logits); all_labels.append(yb)
all_logits = torch.cat(all_logits); all_labels = torch.cat(all_labels).numpy()
probs = torch.softmax(all_logits, dim=1).numpy()
preds = probs.argmax(1)

print("Accuracy :", round(accuracy_score(all_labels, preds), 4))
print("AUC      :", round(roc_auc_score(all_labels, probs[:,1]), 4))
print("\\nConfusion matrix:\\n", confusion_matrix(all_labels, preds))
print("\\n", classification_report(all_labels, preds,
                                   target_names=[id2label[0], id2label[1]]))""")

code("""# 混淆矩陣視覺化
cm = confusion_matrix(all_labels, preds)
fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels([id2label[0], id2label[1]])
ax.set_yticklabels([id2label[0], id2label[1]])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
plt.title("Confusion Matrix"); plt.colorbar(im); plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------------------
md("""## 8. 儲存 LoRA adapter 與推論示範

LoRA 只需儲存極小的 adapter 權重(數 MB),載入時再套回 base model。""")

code("""SAVE_DIR = "vit_lora_nodule"
model.save_pretrained(SAVE_DIR)
print("已儲存 LoRA adapter 至:", SAVE_DIR)

# 重新載入示範
from peft import PeftModel
reload_base = ViTForImageClassification.from_pretrained(
    CKPT, num_labels=2, id2label=id2label, label2id=label2id)
reload_model = PeftModel.from_pretrained(reload_base, SAVE_DIR).to(device).eval()

# 對單一測試樣本推論
x, y = test_ds[0]
with torch.no_grad():
    logit = reload_model(pixel_values=x.unsqueeze(0).to(device)).logits
    p = torch.softmax(logit, 1)[0].cpu()
print(f"真實:{id2label[int(y)]}")
print(f"預測:{id2label[int(p.argmax())]}  (良性={p[0]:.3f}, 惡性={p[1]:.3f})")""")

# ---------------------------------------------------------------------------
md("""## 9. 實驗結果(實跑紀錄)

在本機 **Mac MPS GPU** 上以 8 epochs、batch=16、LoRA(r=8)、lr=5e-4 訓練,
每 epoch 約 50 秒,總計約 7 分鐘。**僅訓練 296,450 / 86,096,644 = 0.34% 參數。**

| 指標 | 測試集數值 |
|---|---|
| **Accuracy** | **0.8581** |
| **AUC** | **0.9223** |
| benign (precision / recall) | 0.92 / 0.90 |
| malignant (precision / recall) | 0.64 / 0.70 |

**混淆矩陣**(列=真實,欄=預測):

|  | 預測 benign | 預測 malignant |
|---|---|---|
| **真實 benign** | 221 | 25 |
| **真實 malignant** | 19 | 45 |

訓練過程(train_loss / val_acc):

```
Epoch 1/8 | loss 0.5990 | val_acc 0.8667
Epoch 2/8 | loss 0.3958 | val_acc 0.8424
Epoch 3/8 | loss 0.3388 | val_acc 0.8364
Epoch 4/8 | loss 0.3102 | val_acc 0.8242
Epoch 5/8 | loss 0.2551 | val_acc 0.7818
Epoch 6/8 | loss 0.2227 | val_acc 0.8545
Epoch 7/8 | loss 0.1722 | val_acc 0.8364
Epoch 8/8 | loss 0.1542 | val_acc 0.8424
```

> 資料類別不平衡(benign 遠多於 malignant),故在損失函數加入 **class weight**,
> 使惡性 recall 提升到 0.70。**AUC 0.92** 顯示模型整體鑑別力良好。

## 小結

- 用 **NoduleMNIST3D**(小型公開醫學影像庫)做 **肺結節良惡性分類**
- 透過 **三正交切片** 把 3D 體積轉成 2D RGB,得以使用 2D **預訓練 ViT**
- 以 **PEFT LoRA** 微調,只訓練 **0.34%** 參數,大幅降低運算與記憶體需求
- 在 Mac MPS GPU 上 8 epochs 約 7 分鐘,達到 **Accuracy 0.858 / AUC 0.922**

**可延伸方向**:改用 `size=64` 較大資料、增加 epoch、嘗試多切片(非只中心)、
或改做其他 MedMNIST 資料集(如 OrganMNIST3D)。
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("NoduleMNIST3D_ViT_LoRA.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook 已產生:NoduleMNIST3D_ViT_LoRA.ipynb,共", len(cells), "個 cell")
