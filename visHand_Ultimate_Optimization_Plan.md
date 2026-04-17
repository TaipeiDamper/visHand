# visHand 性能與感知終極優化藍圖 (v7.Summit_Final)

## 1. 核心理念
本計畫旨在將 visHand 從基礎的視覺工具提升至「工業級」手勢互動系統。核心思維是：**用極少量的高速代數運算（預測、約束、光流），去換取沉重神經網路（MediaPipe）的效能釋放。**

---

## 2. 實作功能矩陣

### A. 結構層：非同步架構 (Infrastructure)
*   **LIFO 多線程讀取與 FramePacket**: 封裝 Image + ID + Timestamp。
    *   `[Programmer's Note]`: 嚴格限制緩衝長度 (`maxsize=2`) 並使用 Deepcopy。
*   **非同步數據總線 (Async Data Bus)**: 
    *   `[Architect's Note]`: 解耦「反射動作」與「軌跡判斷」。

### B. 偵測層：區域邏輯優化 (Detection)
*   **金字塔搜圖 (Pyramid ROI)**: 
    *   搜尋模式 (Global) 與 追蹤模式 (ROI)。
    *   `[Programmer's Note]`: 實作 **Letterboxing**。
*   **運動顯著性掩模 (Saliency Masking)**: 
    *   `[Architect's Note]`: 針對背景雜訊區域進行靜態屏蔽。

### C. 訊號處理層：感知穩定化 (Signal)
*   **座標重映射優先權 (Remapping Priority)**: 
    *   `[Architect's Note]`: **映射 (Remap) -> 約束 (Constraint) -> 平滑 (Filter)**。
*   **骨骼運動學約束 (Kinematic Clamp)**: 
    *   `[Programmer's Note]`: 導入 **「延遲穩定校準」**。
*   **卡爾曼預測項**: 預測 33ms 後的座標位置，補償處理延遲。

### D. 高階技巧：流暢度與穩定性 (Advanced Tricks)
*   **光流預緩衝與回歸穩定**: 
    *   `[Consultant's Wisdom]`: 引入 **適配器模式 (Adapter Pattern)**，確保偵測引擎可隨時無痛替換。
*   **延遲對齊 (Latency Shifting)**: 視覺影音延後，達成點位影像零偏移。

### E. 智慧加速：邏輯節能 (Lazy Evaluation)
*   **影像熵動態跳幀 (Entropy Skipping)**: 像素變化微小時自動跳過訓練。
*   **手勢導向過濾遮罩 (Gesture Masking)**: 僅對 active 點位進行過濾。

---

## 3. 架構師的實作路徑 (Architect's Revised Roadmap)

1.  **Phase 1 (Infrastructure)**: `FramePacket` 線程管道、座標還原映射層。
2.  **Phase 2 (Signal Hygiene)**: **骨骼物理約束**、信心加權過濾、1€ Filter 優化。
3.  **Phase 3 (Optimization & ROI)**: **Letterboxing ROI 裁切**、光流補幀、Auto-Gain。
4.  **Phase D (Brain & State)**: 卡爾曼預測修正、手勢屏蔽遮罩、意圖預判系統。
5.  **Phase E (Final Adaptation)**: 熵增跳幀、硬體曝光調校、動態參數熱加載 (Hot-Reload)。

---

## 4. 系統健壯度對策表 (Architect's Fail-safe)

| 異常風險 | 執行邏輯 | 顧問的頂峰點評 [Consultant's Verdict] |
| :--- | :--- | :--- |
| **ROI 全面脫軌** | 信心 < 0.3 即觸發 **Global Search**。 | 高度分散式系統中，**「自癒能力 (Self-healing)」** 是第一優先級。 |
| **光流累積漂移** | 每秒強制 20 次重置。 | **可觀測性 (Telemetry)** 必須到位，記錄每一幀的累積誤差。 |
| **環境光劇烈變化** | ROI 自動增益。 | AI 的視覺隔離是王道，莫讓增強後的影像干擾使用者視覺。 |

---

## 5. 顧問終極意見 [Consultant's Final Verdict]
1.  **性能觀測器 (Performance Telemetry)**：必須內建 Profiler，監控每一毫秒的去向。
2.  **動態參數調優 (Runtime Tuning)**：所有閾值必須可隨時動態調整，以適應不同感光元件的物理特性。
3.  **透明化設計 (Transparency)**：最好的科技是隱形的。我們追求的是使用者「感知不到技術」，只感覺到自己的手。

---

## 6. 邏輯檢查結論 (Final Master Sign-off)
本計畫集齊了開發、架構、產品、管理與諮詢的所有頂層邏輯。這是一份無懈可擊的終極實作指南。
