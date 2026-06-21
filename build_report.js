const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        ImageRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
        WidthType, ShadingType, TableOfContents, PageNumber, Header, Footer,
        PageBreak } = require("docx");

const CONTENT_W = 9360;
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function cell(text, { w, head = false, bold = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA }, margins: cellMargins,
    shading: head ? { fill: "2E75B6", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: align, children: [
      new TextRun({ text, bold: head || bold, color: head ? "FFFFFF" : "000000" })] })],
  });
}

function table(widths, rows) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((c, ci) => cell(c, {
        w: widths[ci], head: ri === 0,
        align: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      })),
    })),
  });
}

function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120, line: 300 }, ...opts,
    children: [new TextRun({ text, ...opts.run })] });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "b", level: 0 },
    spacing: { after: 60, line: 300 }, children: [new TextRun(text)] });
}
function h(text, level) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}
function img(path, w, hh, caption) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path),
        transformation: { width: w, height: hh },
        altText: { title: caption, description: caption, name: caption } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 20, color: "555555" })] }),
  ];
}

const headingStyle = (id, name, size, lvl) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { size, bold: true, font: "Arial", color: "1F3864" },
  paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: lvl },
});

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 30, 0),
      headingStyle("Heading2", "Heading 2", 25, 1),
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET,
    text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun("第 "), new TextRun({ children: [PageNumber.CURRENT] }), new TextRun(" 頁")] })] }) },
    children: [
      // ---- Title block ----
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 120 },
        children: [new TextRun({ text: "肺結節良／惡性分類", bold: true, size: 48, color: "1F3864" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
        children: [new TextRun({ text: "以 Vision Transformer + LoRA 微調 NoduleMNIST3D", size: 30, color: "2E75B6" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "深度學習實作報告", size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "資料集:NoduleMNIST3D (MedMNIST v2)  ·  模型:ViT-base  ·  微調:PEFT LoRA", size: 20, color: "555555" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1000 },
        children: [new TextRun({ text: "日期:2026 / 06 / 21", size: 20, color: "555555" })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- TOC ----
      h("目錄", HeadingLevel.HEADING_1),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- 1 ----
      h("1. 研究主題與動機", HeadingLevel.HEADING_1),
      p("肺結節（pulmonary nodule）是肺癌早期篩檢的重要徵象。臨床上需依據電腦斷層（CT）影像判斷結節為良性或惡性，但人工判讀耗時且高度依賴經驗。本報告以深度學習方法，建立一個自動分類肺結節良／惡性的模型，作為輔助診斷的概念驗證。"),
      p("考量運算資源與作業規模，本研究選用小型公開資料庫 NoduleMNIST3D（約 1,600 筆樣本），並採用「預訓練模型 + 參數高效微調（PEFT）」的策略，使整個訓練流程能在一般筆電的 GPU（Apple Silicon MPS）上於數分鐘內完成。"),

      // ---- 2 ----
      h("2. 資料集介紹:NoduleMNIST3D", HeadingLevel.HEADING_1),
      p("NoduleMNIST3D 為 MedMNIST v2 標準化醫學影像資料集之一，源自 LIDC-IDRI 肺部 CT 公開資料庫，將結節整理成統一的 28×28×28 三維體積（voxel）。任務為二分類:良性（benign）與惡性（malignant）。"),
      table([3120, 3120, 3120], [
        ["資料分割", "樣本數", "說明"],
        ["訓練 (train)", "1,158", "用於模型訓練"],
        ["驗證 (val)", "165", "用於監控過擬合"],
        ["測試 (test)", "310", "最終評估（benign 246 / malignant 64）"],
      ]),
      p("", { spacing: { after: 60 } }),
      p("資料特性:影像為單通道灰階、值域 0–255、尺寸 28³（MNIST-like 最小版本）。測試集中良性樣本約為惡性的 4 倍，屬於典型的「類別不平衡」資料，這在後續方法與結果中都需特別處理。"),
      ...img("fig_samples.png", 300, 550, "圖 1. NoduleMNIST3D 測試樣本（每列為一結節的三正交切片;上 3 列良性、下 3 列惡性）"),

      // ---- 3 ----
      h("3. 方法", HeadingLevel.HEADING_1),
      h("3.1 3D → 2D 前處理（三正交切片）", HeadingLevel.HEADING_2),
      p("Vision Transformer 是針對 2D 影像設計的模型，而資料為 3D 體積。為了能直接利用 2D 影像的大規模預訓練權重，本研究將每個體積取三個正交方向的中心切片:"),
      bullet("Axial（軸狀面）— 沿深度軸的中央切片"),
      bullet("Coronal（冠狀面）— 沿高度軸的中央切片"),
      bullet("Sagittal（矢狀面）— 沿寬度軸的中央切片"),
      p("三張灰階切片分別作為 R、G、B 三個通道，組成一張 RGB 影像，再以雙線性內插放大到 224×224，並以 mean=0.5、std=0.5 正規化。此設計兼顧兩點:可沿用 ImageNet 預訓練權重，同時保留 3D 結構的多視角資訊。"),

      h("3.2 模型架構:Vision Transformer", HeadingLevel.HEADING_2),
      p("採用 HuggingFace 的 google/vit-base-patch16-224-in21k（ViT-base，於 ImageNet-21k 預訓練）。模型將 224×224 影像切成 16×16 的 patch，經線性投影後加上位置編碼，送入 12 層 Transformer Encoder，最後以 [CLS] token 接一個全新初始化的 2 類分類頭。"),

      h("3.3 參數高效微調:PEFT LoRA", HeadingLevel.HEADING_2),
      p("LoRA（Low-Rank Adaptation）在凍結原始權重的前提下，於注意力層的 query、value 線性層注入低秩矩陣（rank=8, alpha=16）做為可訓練的旁路；分類頭因為是新初始化，故設為完整可訓練。如此只需更新極少量參數，即可將大型預訓練模型調適到新任務。"),
      table([4680, 2340, 2340], [
        ["項目", "數值", "佔比"],
        ["可訓練參數", "296,450", "0.34%"],
        ["模型總參數", "86,096,644", "100%"],
      ]),

      // ---- 4 ----
      h("4. 實驗設定", HeadingLevel.HEADING_1),
      table([4680, 4680], [
        ["超參數 / 環境", "設定"],
        ["運算裝置", "Apple Silicon GPU（PyTorch MPS 後端）"],
        ["優化器", "AdamW（lr=5e-4, weight_decay=1e-4）"],
        ["損失函數", "CrossEntropy + class weight（處理不平衡）"],
        ["Batch size", "16"],
        ["訓練輪數", "8 epochs（每 epoch 約 50 秒，總計約 7 分鐘）"],
        ["資料擴增", "隨機水平／垂直翻轉"],
        ["LoRA 設定", "r=8, alpha=16, dropout=0.1, target=query/value"],
      ]),

      // ---- 5 ----
      h("5. 實驗結果", HeadingLevel.HEADING_1),
      p("模型在測試集上達到 Accuracy 0.858、AUC 0.922。各項指標如下表:"),
      table([4680, 2340, 2340], [
        ["指標", "數值", "備註"],
        ["Accuracy", "0.8581", "整體正確率"],
        ["AUC", "0.9223", "ROC 曲線下面積（鑑別力）"],
        ["benign  precision / recall", "0.92 / 0.90", "良性"],
        ["malignant precision / recall", "0.64 / 0.70", "惡性"],
      ]),
      p("", { spacing: { after: 60 } }),
      ...img("fig_training.png", 460, 276, "圖 2. 訓練過程:train loss 持續下降、val accuracy 大致穩定"),
      ...img("fig_confusion.png", 360, 326, "圖 3. 測試集混淆矩陣（良性 221 正確、惡性 45 正確）"),
      p("下圖為模型在測試樣本上的實際預測。模型對良性與明顯惡性結節判斷準確且高信心;主要失誤集中在較小、較不明顯的惡性結節，與惡性 recall 0.70 的量化結果一致。"),
      ...img("fig_demo_predictions.png", 540, 280, "圖 4. 預測示範:上排良性、下排惡性（綠=答對、紅=答錯，括號為信心值）"),

      // ---- 6 ----
      h("6. 討論", HeadingLevel.HEADING_1),
      bullet("效率:LoRA 僅訓練 0.34% 的參數即達到 AUC 0.92，顯示參數高效微調在小型醫學影像任務上的明顯優勢——大幅降低記憶體與運算需求，使一般筆電 GPU 即可訓練。"),
      bullet("收斂:train loss 從 0.60 穩定下降至 0.15，而 val accuracy 在第 1 個 epoch 後僅小幅震盪，顯示模型很快收斂；再增加輪數容易過擬合，故 8 epochs 為本設定的合適選擇。"),
      bullet("類別不平衡:測試集惡性樣本僅佔約 21%。雖已加入 class weight 將惡性 recall 提升至 0.70，但惡性 precision（0.64）仍偏低，是此資料的主要瓶頸。"),
      bullet("臨床意涵:在篩檢場景中，漏掉惡性（false negative）代價最高，因此惡性 recall 是關鍵指標；本模型 19 例惡性被誤判為良性，仍有改善空間。"),

      // ---- 7 ----
      h("7. 結論與未來方向", HeadingLevel.HEADING_1),
      p("本研究成功以「三正交切片 + 預訓練 ViT + LoRA 微調」的流程，在小型 3D 醫學影像資料 NoduleMNIST3D 上完成肺結節良／惡性自動分類，於測試集達到 Accuracy 0.858、AUC 0.922，且整體訓練可在筆電 GPU 上數分鐘內完成。"),
      p("未來可朝以下方向延伸:"),
      bullet("使用更高解析度版本（size=64 或 128）以保留更多細節"),
      bullet("採用多切片或真正的 3D ViT 以完整利用體積資訊"),
      bullet("以 Focal Loss 或過採樣進一步改善惡性類別的偵測"),
      bullet("套用至其他 MedMNIST 3D 資料集（如 OrganMNIST3D）驗證泛用性"),

      // ---- Appendix ----
      h("附錄:實作與資源", HeadingLevel.HEADING_1),
      bullet("程式碼:NoduleMNIST3D_ViT_LoRA.ipynb（可於本機 MPS 或 Colab GPU 執行）"),
      bullet("資料集:MedMNIST v2 — https://medmnist.com/"),
      bullet("預訓練模型:google/vit-base-patch16-224-in21k（HuggingFace）"),
      bullet("套件:PyTorch、transformers、peft、medmnist、scikit-learn"),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("肺結節分類_ViT_LoRA_報告.docx", buf);
  console.log("報告已產生:肺結節分類_ViT_LoRA_報告.docx");
});
