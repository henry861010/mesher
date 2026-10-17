# circular.imprint_v2 大半徑空洞案例

這份案例集對應 **circle radius 遠大於 element size，且局部網格可能很細** 的使用場景。
先前十組「小圓配粗網格」案例與結果已全部替換；目前 `case_01`～`case_10` 都是新的案例，
請以現在的參數與報告對照元素／節點編號。

## 尺度與目前 process_flow 設定

十組都使用半徑 **300000**、目標格距 **5000**，比值 **60**。
實際生成的 X／Y 最大格距也不超過 5000；局部細格距保留在約 0.184～159.054。
所有初始網格都是涵蓋**整個圓周**的矩形網格，範圍為 `[-320000, 320000]²`，
不是把小圓案例等比例放大，也不是只保留圓周附近的裁剪網格。
初始網格包含圓的極值線 `±300000` 與 process_flow 的支撐線 `±320000`
（`band_width=2h`，支撐半徑為 `R+2*band_width=R+4h`）。

容差直接對齊目前的 [`process_flow/pipeline.py`](../../../src/mesher/process_flow/pipeline.py)：

| 參數 | process_flow 計算方式 | 本案例傳入值 |
| --- | --- | ---: |
| `projection_tolerance` | `0.2 * h` | 1000 |
| `guide_tolerance` | `0.2 * h` | 1000 |
| `merge_tolerance` | `0.002 * h` | 10 |
| `minimum_area` | `0.0004 * h²` | 10000 |
| `coordinate_tolerance` | 一般無 KOZ 細化時的預設值 | 0.001 |

目前 [`imprint_v2/mesher.py`](../../../src/mesher/mesh2d/circular/imprint_v2/mesher.py)
會執行 `minimum_area = min(1, minimum_area)`，因此實際清理面積門檻是 **1**。
案例保留傳入的 **10000**，報告另記錄 requested／effective 值，不改動生產演算法。
這些是對齊 process_flow 參數、網格尺度與支撐線的 **imprint API 反例**；
不把它們宣稱為完整 Standard V1 geometry → 3D pipeline 的端到端案例。

## 十組案例與判定標準

| 案例 | 初始元素數 | 最小格距 | 最大格距 | 內部空洞數 | 內部空洞面積 |
| --- | ---: | ---: | ---: | ---: | ---: |
| case_01 | 18495 | 20.00000 | 5000 | 1 | 419370.112 |
| case_02 | 18224 | 100.00000 | 5000 | 1 | 823589.544 |
| case_03 | 18225 | 0.18387 | 5000 | 1 | 334839.753 |
| case_04 | 18090 | 20.00000 | 5000 | 2 | 1177545.490 |
| case_05 | 18496 | 50.00000 | 5000 | 1 | 445702.676 |
| case_06 | 18088 | 20.00000 | 5000 | 1 | 504350.919 |
| case_07 | 18360 | 50.00000 | 5000 | 1 | 466102.136 |
| case_08 | 18496 | 159.05414 | 5000 | 1 | 483275.602 |
| case_09 | 18225 | 151.48946 | 5000 | 1 | 515541.489 |
| case_10 | 18360 | 20.00000 | 5000 | 1 | 590740.098 |

每一組都必須同時滿足：

- 初始網格有效，沒有缺失面積，且所有實際格距不超過 5000。
- 使用上述原始 process_flow 參數，公開 `imprint_circle` 正常返回。
- 投影後仍完整覆蓋原矩形，但已出現無效元素。
- 清理後首次出現正面積內部空洞；後續切割沒有補回缺失區域。
- 保留其他設定，關閉投影或移除 guide 後，公開 API 成功並恢復完整覆蓋。

所有案例的最早無效階段都是 `projection`，最早空洞階段都是 `cleanup`。
這十組來自不同的固定亂數 seed，再嵌回完整圓網格驗證；不是同一個例子的十種旋轉或縮放。
`source_seed` 僅供探索來源追溯，完整座標已明確列在 [`cases.json`](cases.json)，重現不使用亂數。

## 目前找到的失效路徑

```text
完整的細分矩形網格
    ↓ _to_circle：徑向候選選擇＋guide 約束／目標合併
節點跨過相鄰細格線，局部元素折返或自交
    ↓ _remove_redundant_element → _repair_elements
無法以目前對角線規則修復的元素被丟棄
    ↓ _imprint_circle
剩下的有效元素正常切割，空洞保留，公開 API 成功返回
```

這次十組都有 guide；在這些案例裡，名目上只有 `0.2*h` 的容差仍可能大於**局部格距**。
`_to_circle` 只以徑向差選候選點，guide 則以距離選約束，沒有檢查最終移動是否折疊鄰接元素。
因此「圓半徑遠大於元素」本身無法避免這條失效路徑。
不能把之前小半徑、無 guide 的圓周目標合併案例直接套用到本案例集。

具體例子：`case_01` 的節點 **3696** 從
`[-214046.54157, -211459.34966]` 移到
`[-212355.12904092157, -211908.7048]`，位移約 **1750.085**。
這大於徑向候選容差 1000，也大於部分局部格距；元素 **3534** 隨後在修復時被丟棄，
留下約 **419370.112** 的內部空洞。元素編號均從零開始，對應清理前的 `projection.npz`。

`_repair_elements` 只接受其中一條對角線能形成兩個正面積三角形的非凸 Quad4；
若兩種切法都不成立，就不輸出該元素。它沒有重建被刪除的區域，也沒有檢查全域覆蓋。
公開 wrapper 的交易性只處理例外，未對成功輸出做整體覆蓋檢查，因此仍會正常提交空洞 mesh。
各組的對照實驗與清理列號請見各自的 `report.json`。

## 互動式前後對照

從 repository 根目錄執行；不需設定 `PYTHONPATH`。
需要現有的 visualization 額外依賴，尚未安裝時可使用 `python -m pip install '.[visualization]'`。

```bash
venv/bin/python examples/failures/imprint_v2_holes/view.py --case case_01
venv/bin/python examples/failures/imprint_v2_holes/view.py --case 3
venv/bin/python examples/failures/imprint_v2_holes/view.py --list
```

[`view.py`](view.py) 左右顯示 Before／After，兩側同步縮放、平移並固定 XY 俯視角。
為了讓大圓上的小空洞可見，啟動時會對準 `inspection_bounds` 記錄的失敗區域。

- 滾輪／右鍵拖曳：縮放。
- 左鍵拖曳：平移。
- `R`：顯示完整矩形網格與整個圓。
- `F`：回到案例的失敗區域。

mesh 為淺藍色、元素邊線為深色、參考圓為藍色、guide 為橘色；內部白色區域可用來觀察空洞。
參考圓取樣精度會依最小格距增加，避免大半徑的粗圓周折線在細格網附近造成誤判。
每次執行都從參數生成獨立的前後 mesh，實際呼叫公開 `circular.imprint_v2.imprint_circle`，
不使用靜態 mesh 快照，也不執行 Shapely 診斷。

## 診斷、快照與資料欄位

```bash
MPLCONFIGDIR=/tmp/mesher-matplotlib PYTHONPATH=src venv/bin/python \
  examples/failures/imprint_v2_holes/diagnose.py --verify --plot

# 單獨重現，不覆寫現有結果
PYTHONPATH=src venv/bin/python examples/failures/imprint_v2_holes/diagnose.py \
  --case case_03 --output /tmp/imprint_v2_case_03 --verify --plot
```

診斷需要 NumPy／Shapely；`--plot` 另外需要 Matplotlib。
完整圓網格每組約 1.8 萬個元素，重算十組及其對照實驗需要數分鐘。
[`results/overview.png`](results/overview.png) 與各組 `stages.png` 顯示失敗區域的局部放大圖，
底層 `.npz` 仍保存完整網格。

`cases.json` 每組保留以下欄位：

- `grid_x_list`、`y_list`、`grid_circler_center_coordinate`、`circle_radius`。
- `target_edge_size`、`coordinate_tolerance`、四個 imprint 容差／面積參數與 `guide_segments`。
- `mesh_profile`：半徑與目標尺寸比、實際最小／最大格距、初始元素數。
- `inspection_bounds`：局部檢查框 `[xmin, ymin, xmax, ymax]`。

其中 `grid_circler_center_coordinate` 對應 API 的 `center`，`circle_radius` 對應 `radius`。
`results/case_XX/` 保存：

- `original.npz`、`projection.npz`、`cleanup.npz`、`split.npz`：四階段完整 `nodes`／`elements`。
- `report.json`：各階段缺失／重疊面積、無效元素、移動節點、實際面積門檻與清理輸出。
- `missing.geojson`：最終缺失區域的實際幾何。
- `stages.png`：逐階段局部圖，紫色數字為無效元素列號。

報告中的對照為：關閉投影、移除 guides、停用圓周目標合併、將面積門檻設為零。
十組將面積門檻設為零後，正面積內部空洞仍然存在。
這些對照用來辨別因素，不代表建議修改使用者的生產參數。
[`results/report.json`](results/report.json) 彙整十組結果與使用的 Python／NumPy／Shapely 版本。

## 量測及修復後驗證

缺失面積以「原始矩形減去元素 footprint 聯集」量測，區分內部空洞與接觸外框的缺口。
元素自交時，只在量測副本上使用 Shapely `make_valid` 以量測各葉瓣；
無效元素另外列出，生產 mesh 不會被這個量測修復。
封閉缺失區域必須不接觸外框，且面積大於 `domain_area * 1e-10`，本案例為 **40.96**。
此閾值只用來排除浮點碎片，不是 imprint 的面積門檻。

`--verify` 檢查尺度與初始覆蓋、首次失效階段、逐階段執行與公開 API 一致、
最終封閉空洞、split 沒有改變缺失面積，以及關閉觸發因素的對照恢復覆蓋。
一般 unittest 另外確認案例參數符合目前 process_flow 常數、格距限制與互動檢視器行為。
修復演算法後，已知失敗特徵檢查預期會失敗，屆時應將案例轉成無空洞的回歸測試。
