# circular.imprint_v2 大半徑案例檢查

這份案例集對應 **circle radius 遠大於 element size，且局部網格可能很細** 的使用場景。
`cases.json`、互動檢視器與診斷結果都使用目前主流程的參數。
原有 case_01～case_10 完整保留；2026-10-18 新增 case_11～case_20，保存其修正前的內部空洞重現資料。

## 尺度與目前參數

二十組都使用半徑 **300000**、平面目標尺寸 **5000**，比值 **60**。
初始網格涵蓋整個圓周，範圍為 `[-320000, 320000]²`，所有實際格距不超過 5000，
局部細格距最小值介於約 0.01～159.054。網格包含圓的極值線 `±300000` 與支撐線 `±320000`
（`band_width=2h`，支撐半徑為 `R+2*band_width=R+4h`）。

參數對齊 [`mesh25d/plan2d.py`](../../../src/mesher/mesh25d/plan2d.py)：

| 參數 | 目前計算方式 | 本案例傳入值 |
| --- | --- | ---: |
| `projection_tolerance` | `0.2 * h` | 1000 |
| `guide_tolerance` | `max(0.001 * h, 1)` | 5 |
| `merge_tolerance` | `0.002 * h` | 10 |
| `minimum_area` | `0.0004 * h²` | 10000 |
| `coordinate_tolerance` | 本案例的座標容差 | 0.001 |

`guide_tolerance` 的目的是「判斷 node 是否在 guide segment 上面；而非用來找 segment 附近 node」。
其單位與模型座標相同。
目前 [`imprint_v2/mesher.py`](../../../src/mesher/mesh2d/circular/imprint_v2/mesher.py)
仍執行 `minimum_area = min(1, minimum_area)`，實際清理面積門檻因此是 **1**；報告同時記錄傳入與生效值。

這些是 imprint API 層級的檢查案例，並非完整 geometry → 3D pipeline 的端到端案例。
`source_seed` 僅供探索來源追溯，完整座標已列在 [`cases.json`](cases.json)，執行不使用亂數。

## 圓周局部處理與固定節點

公開 `imprint_circle` 會分批用元素 AABB 的徑向距離區間，選出與
`R ± (projection_tolerance + merge_tolerance + 數值容差)` 相交的元素，
再加入共享節點的相鄰元素。不是取整個圓內，也不是只檢查頂點是否靠近圓周；
大元素的邊穿過圓周時仍會納入。

節點到元素的 CSR 索引與合併分桶每次操作各建立一次。局部區域沿實際等價群組、
共享節點及切割邊的所有 owner 擴張，直到回填不會影響區域外的連接關係。
等價群組保留最低原始全域 node ID 的 XYZ。三個幾何階段使用緊密的局部編號，
最後依明確的節點映射與元素來源回填並壓縮一次；不以座標猜測原始 ID。
因此圈外幾何與拓撲保留，但最終 ID 可能因壓縮改變。沒有候選元素時直接返回。
操作失敗不會修改輸入 mesh。

沿 guide 的投影，在目標合併前檢查路徑上的實際 mesh node：若嚴格介於原點與
投影目標之間有水平、垂直 guide 共同鎖定且依現有約束無法移動的節點，
該候選保留原座標並退出圓周目標合併。允許抵達圓上的固定點。
guide 路徑可能超出初始環帶；這些路徑使用原始全域座標查詢唯讀約束節點，
不會順帶清理整條 guide 附近的網格。主流程使用 `max(0.001 * h, 1)`，
本案例 `h=5000`，因此傳入的絕對距離容差為 5。

自由節點的徑向投影也會在目標合併前檢查：原座標至徑向圓周交點的開放路徑，
先將 guide segment 的兩端各延長 `projection_tolerance`，再檢查途中是否與其相交
（包含碰到延長後的 guide 端點與共線重疊）。若相交，
該節點保留原 XYZ 並退出此次目標合併。不要求交點存在 mesh node，也不要求 guide 與圓相交。
交點是否落在延長後的有限 guide 上使用 `max(guide_tolerance, guide 數值容差)`；
本案例每端延長 1000，端點容差 5 另計。延長量只用於原始徑向路徑檢查，
不修改共享 guide 資料；節點所屬 guide、guide 與圓的投影交點及沿 guide 移動仍依原始有限 segment。
路徑起終點只以數值容差排除，因此允許目標剛好抵達 guide。平行且不相交的路徑不受影響。
此檢查只針對原始徑向路徑，合併後的路徑不重新檢查。

case_12 的 **17751** 原本從 `[65574.67161, 293757.47317]` 徑向移往
`[65359.52425331458, 292793.6689707965]`，途中穿過 `y=293657.47317` 的水平 guide。
加入徑向穿越保護後，17751 保持原位，元素 17486 不再自交。重新診斷為 **pass**：
最終空洞與缺失面積均為 0，重疊面積約 0.000671，低於量測容差 40.96。
投影階段仍有退化元素，經 cleanup 後最終無無效元素；公開 API 與逐階段重播一致。
本次重播輸出至暫存目錄，原有 `results/` 保留修正前快照。

case_16 的 **10277** 原本從 `[299129.09297, 30235.92041]` 徑向移往
`[298479.07468743593, 30170.21666697216]`，交點在垂直 guide 端點外約 **10.108**，
超過原本端點容差 5。每端延長 1000 後，10277 保持原位，元素 10203 不再自交；
重播為 **pass**，最終空洞、缺失及重疊面積均為 0。
case_13 的合併後穿線與 case_14 的沿 guide 移動未納入這次修改，仍有空洞。

case_04 的 **12497** 原本會從 x=273748.32777 移到 272652.7079508164，
跨過固定的 **12496**（x=273498.32777），使元素 12270、12405 自交而被清理移除。
現在 12496、12497 都保持原位，最終空洞與重疊面積均為零。
12498、12499 的徑向距離差原本就大於投影容差 1000，並未被投影。

## 保存的診斷基準（徑向穿越保護前）

| 案例 | 保存狀態 | 初始元素數 | 最小格距 | 內部空洞數 | 內部空洞面積 | 重疊面積 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| case_01 | pass | 18495 | 20.00000 | 0 | 0.000 | 0.000 |
| case_02 | pass | 18224 | 100.00000 | 0 | 0.000 | 0.000 |
| case_03 | pass | 18225 | 0.18387 | 0 | 0.000 | 0.000 |
| case_04 | pass | 18090 | 20.00000 | 0 | 0.000 | 0.000 |
| case_05 | pass | 18496 | 50.00000 | 0 | 0.000 | 0.000 |
| case_06 | pass | 18088 | 20.00000 | 0 | 0.000 | 0.000 |
| case_07 | pass | 18360 | 50.00000 | 0 | 0.000 | 0.000 |
| case_08 | pass | 18496 | 159.05414 | 0 | 0.000 | 0.000 |
| case_09 | pass | 18225 | 151.48946 | 0 | 0.000 | 0.000 |
| case_10 | pass | 18360 | 20.00000 | 0 | 0.000 | 0.000 |
| case_11 | fail | 18224 | 0.10000 | 2 | 56703.770 | 2218239.535 |
| case_12 | fail | 18496 | 0.10000 | 1 | 26335.956 | 3691639.969 |
| case_13 | fail | 18904 | 1.00000 | 1 | 948529.302 | 6074.907 |
| case_14 | fail | 18225 | 5.00000 | 2 | 2180.989 | 1257775.500 |
| case_15 | fail | 18904 | 0.10000 | 1 | 54239.893 | 902385.878 |
| case_16 | fail | 18492 | 50.00000 | 1 | 1150391.499 | 2311444.631 |
| case_17 | fail | 18360 | 0.10000 | 1 | 104764.050 | 1068658.723 |
| case_18 | fail | 18487 | 0.01000 | 1 | 178238.295 | 39436233.077 |
| case_19 | fail | 18492 | 0.01000 | 1 | 14033.609 | 3266008.135 |
| case_20 | fail | 18765 | 5.00000 | 1 | 420877.667 | 1275224.209 |

上表對應 [`results/report.json`](results/report.json) 保存的修正前資料；以目前程式重播可能得到不同結果。
`pass` 表示最終網格沒有無效元素，缺失、重疊與超出原始範圍的面積均未超過量測容差；
其餘標為 `fail`。投影階段的無效元素另列在報告中，不直接代表最終結果失敗。

case_01 的節點 **3696** 現在從 `[-214046.54157, -211459.34966]`
移到徑向投影點 `[-213417.91003762258, -210838.31643032352]`，位移約 **883.663**。
它離 guide 約 449.355，大於目前的容差 5，因此不受該 guide 約束。
節點／元素編號均從零開始；清理會重新編號，移動前後請對照 `original.npz` 與 `projection.npz`。

## 2026-10-18 新增案例

以固定亂數種子 100000～100170 探索 171 組局部細格線與有限長度的水平／垂直 guides，
先篩選圓周局部網格，再將候選擴充成 `[-320000, 320000]²` 完整網格，
使用公開 `imprint_circle` 重新確認；取得十組後即停止搜尋。
新案例的完整座標、guides、來源種子與發現時量測值都保存於 `cases.json`，執行不需要亂數。

十組初始網格均完整覆蓋矩形，公開 API 與逐階段重播結果完全一致，`--verify` 全數通過。
修正前保存的結果合計有 **12 個內部空洞**；各組空洞面積約為 **2180.989～1150391.499**，
均超過面積容差 **40.96**，且沒有接觸外框。最終元素雖都有效，仍同時存在缺失與重疊。

這十組都在 projection 階段出現自交元素，cleanup 後首次出現空洞，split 階段保留相同缺失面積。
對照實驗中，移除 guides 後十組均無空洞；將面積門檻設為零仍保留原本空洞。
停用圓周目標合併後，只有 case_18 的空洞消失，但該組仍有重疊。
這些修正前結果將調查範圍指向 guide 約束投影與細格網的互動。
部分關閉投影的對照會拋出 `imprinted child areas do not cover their parent`，也已記錄於各組報告。

可直接檢視新案例：

```bash
venv/bin/python examples/failures/imprint_v2_holes/view.py --case case_11
venv/bin/python examples/failures/imprint_v2_holes/view.py --case case_20

# 重算單一新案例，另存結果
MPLCONFIGDIR=/tmp/mesher-matplotlib PYTHONPATH=src venv/bin/python \
  examples/failures/imprint_v2_holes/diagnose.py --case case_11 \
  --output /tmp/imprint_v2_case_11 --verify --plot
```

## 互動式前後對照

從 repository 根目錄執行，不需設定 `PYTHONPATH`。
需要 visualization 額外依賴，尚未安裝時可使用 `python -m pip install '.[visualization]'`。

```bash
venv/bin/python examples/failures/imprint_v2_holes/view.py --case case_01
venv/bin/python examples/failures/imprint_v2_holes/view.py --case 3
venv/bin/python examples/failures/imprint_v2_holes/view.py --list
```

[`view.py`](view.py) 每次從目前案例參數生成獨立的 Before／After mesh，實際呼叫公開
`circular.imprint_v2.imprint_circle`，不讀取靜態快照，也不執行 Shapely 診斷。
兩側同步縮放、平移，固定 XY 俯視角，啟動時對準 `inspection_bounds` 記錄的檢查區域。

- 滾輪／右鍵拖曳：縮放。
- 左鍵拖曳：平移。
- `R`：顯示完整矩形網格與整個圓。
- `F`：回到案例檢查區域。

mesh 為淺藍色，元素邊線為深色，參考圓為藍色，guide 為橘色。
參考圓取樣精度依最小格距增加，避免大半徑的粗圓周折線在細格網附近造成誤判。

## 診斷、快照與驗證

```bash
MPLCONFIGDIR=/tmp/mesher-matplotlib PYTHONPATH=src venv/bin/python \
  examples/failures/imprint_v2_holes/diagnose.py --verify --plot

# 單獨檢查並將結果寫到另一個目錄
PYTHONPATH=src venv/bin/python examples/failures/imprint_v2_holes/diagnose.py \
  --case case_03 --output /tmp/imprint_v2_case_03 --verify --plot
```

診斷需要 NumPy／Shapely，`--plot` 另外需要 Matplotlib。
每組約 1.8 萬個元素，重算全部案例及對照實驗需要數分鐘。
`results/overview.png` 列出全部二十組，各組 `stages.png` 顯示檢查區域的局部放大圖，底層快照保存完整網格。

`results/case_XX/` 保存：

- `original.npz`、`projection.npz`、`cleanup.npz`、`split.npz`：四階段完整 `nodes`／`elements`。
- `report.json`：目前 pass／fail 狀態、參數、各階段缺失／重疊面積、無效元素、移動節點與清理結果。
- `missing.geojson`：最終缺失區域的實際幾何。
- `stages.png`：逐階段局部圖，紫色數字為無效元素列號。

`cases.json` 的 `grid_circler_center_coordinate` 對應 API 的 `center`，`circle_radius` 對應 `radius`。
`mesh_profile` 記錄初始格距與元素數，`inspection_bounds` 是檢查框 `[xmin, ymin, xmax, ymax]`。
報告的 `local_patch` 記錄初始環帶與安全擴張後的元素／節點數；
也記錄關閉投影、移除 guides、停用圓周目標合併及將面積門檻設為零的對照結果。
四階段快照及 cleanup trace 使用與公開 API 相同的局部流程與實際來源映射。

缺失面積以「原始矩形減去元素 footprint 聯集」量測，區分內部空洞與接觸外框的缺口。
元素自交時，只在量測副本使用 Shapely `make_valid`，生產 mesh 不會被這個量測修復。
量測面積容差為 `max(1, domain_area) * 1e-10`，本案例為 **40.96**，用來排除浮點碎片。

`--verify` 檢查初始覆蓋、網格尺度、逐階段執行與公開 API 一致，以及切割前後的缺失面積一致。
它不要求空洞存在；案例的網格品質由報告中的 `status` 表示。
一般 unittest 另確認案例參數符合目前主流程、互動檢視器行為，以及 case_01 的 3696 徑向投影結果。

## 十萬元素完整流程 benchmark

```bash
venv/bin/python examples/failures/imprint_v2_holes/benchmark.py --verify \
  --output /tmp/imprint_v2_benchmark.json
```

[`benchmark.py`](benchmark.py) 比較原本三階段全域參考流程與公開 API 共用的局部引擎。
兩側皆排除輸入網格生成，但計入操作內部的複製、篩選、索引、投影、清理、切割、
回填及壓縮。暖身一次後取五次中位數，另用獨立程序量測操作期間的 traced peak
allocations（含 NumPy；不是整個程序 RSS）。結果會比對與編號無關的元素幾何 digest。

目前數據保存在 [`results/benchmark.json`](results/benchmark.json)。

在本機 Python 3.13.5／NumPy 2.5.2 的五次中位數：

| 指標 | 全域參考流程 | 局部流程 |
| --- | ---: | ---: |
| 完整耗時 | 2.896 秒 | 0.139 秒 |
| 操作 traced peak allocations | 131.3 MiB | 59.5 MiB |

完整流程加速 **20.8 倍**，幾何 digest 一致。

99,856 個元素的網格中，局部區域為 **1,604 個元素／2,084 個節點**。
全域網格增加為 50,176、99,856、200,704 個元素時，幾何處理區域仍保持相同大小。
`--verify` 要求完整流程至少加速 3 倍、局部區域低於全域 5%、參考幾何一致，
且增加遠處網格不增加局部幾何處理量。實際時間仍會隨全域篩選、索引與回填成本成長。
若等價群組確實涵蓋全網格，安全擴張可以退化為全域處理。
