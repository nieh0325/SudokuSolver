# Sudoku Image Solver (圖片數獨解題器)

這個專案是一個基於 Python 的工具，它可以讀取數獨的手機截圖或圖片，透過光學字元辨識 (OCR) 技術自動萃取數獨網格和數字，解開謎題後，將紅色的答案直接繪製回原圖片並儲存。

## 環境需求

此專案需要 Python 3 環境，以及幾項系統與 Python 依賴套件。

### 系統套件 (以 Ubuntu/Debian 為例)
需要安裝 Python 虛擬環境套件與 Tesseract OCR 引擎：
```bash
sudo apt-get update
sudo apt-get install -y python3-venv tesseract-ocr tesseract-ocr-eng
```

### Python 套件
*   `opencv-python`: 用於影像處理與網格萃取。
*   `pytesseract`: 用於數字的光學字元辨識。
*   `numpy`: 用於矩陣運算與影像操作。

## 安裝步驟

1. 建立 Python 虛擬環境並啟動：
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. 安裝必要的 Python 套件：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1.  **準備題目**：將所有想要解開的數獨圖片（.png, .jpg, .jpeg 等）放到 `題目` 資料夾內。
2.  **執行解題**：在終端機中透過虛擬環境執行 `main.py` 即可。

```bash
# 請確保您已在虛擬環境中 (終端機開頭有 `(venv)`)
source venv/bin/activate

# 自動讀取「題目」資料夾的圖片並輸出至「答案」資料夾
python main.py
```

### 執行結果
執行後，程式會自動迭代 `題目` 目錄內的每張圖片，它會處理影像萃取、辨識及解題，最終將包含答案的新圖片以相同檔名儲存至 `答案` 目錄之中。

## 專案結構

```
SudokuSolver/
├── main.py              # 主程式入口
├── src/                 # 核心模組
│   ├── __init__.py
│   ├── extractor.py     # OCR 與網格萃取
│   ├── solver.py        # 回溯法數獨求解器
│   └── renderer.py      # 答案渲染回原圖
├── tools/               # 除錯工具腳本
├── requirements.txt     # Python 依賴清單
├── .gitignore
├── 題目/                # 輸入：數獨截圖
└── 答案/                # 輸出：含答案的圖片
```

*   `main.py`: 專案的主程式入口，負責協調影像萃取、辨識、解題與渲染的整個流程。
*   `src/extractor.py`: 負責影像的前處理、找出數獨網格輪廓，並呼叫 Tesseract 辨識每個格子內的數字。
*   `src/solver.py`: 實作了回溯法 (Backtracking) 演算法來解開 9x9 的數獨謎題。
*   `src/renderer.py`: 負責將解題器算出的答案重新寫回圖片中，精準疊加回原圖上。
*   `tools/`: 開發過程中使用的除錯工具腳本。
*   `requirements.txt`: 紀錄 Python 相關的第三方依賴套件清單。
