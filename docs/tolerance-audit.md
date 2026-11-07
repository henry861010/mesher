# 全專案 0.001 幾何容差盤點與修正計畫

2026-11-07。需求已確認：以模型座標單位的 **絕對距離 0.001** 判定接觸／近似重合。Process-flow 契約及 case1 的單位為 `um`，因此此處是 `0.001 μm`，不是相對誤差、角度或三位小數格式。

**目前專案不具備這項一致性保證。** 問題包含容差未傳遞、相同數值具有不同語意、精確座標鍵保留浮點殘差，以及規劃、修復、驗證各自使用不同門檻。不能用搜尋取代把所有小常數改成 `0.001`；那會把長度與面積、角度、數值誤差混用，甚至掩蓋重疊或反轉元素。

本次只新增調查文件、診斷程式及證據，沒有修改 production 演算法或既有測試。分析依據是目前工作目錄，包括原先尚未提交的修改及新檔。

## Case1 的直接原因

輸入為 [model.geometry.json](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/model.geometry.json) 及 [model.mesh-control.json](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/model.mesh-control.json)。原始失敗 log 位於 [case1 export log](/Users/henry/Downloads/case1/cdb_75eb29e6d4e14728b283ac58b659d96c.log)。

錯誤發生於 `3D_REGION adapter`，`controls[7] ('Die top corners merged 3')`，最後一層的 `boundary_index=2`、`outline_o` 的 `edge_index=9`。錯誤文字同時涵蓋「零長度」與「非水平／垂直」；本案觸發的是**極短邊**。

global grid 的 `hole_2` 留下了以下相鄰錨點：

| 邊 | 起點 | 終點 | 實際長度 μm |
|---|---|---|---|
| 9 | `(2800.0000000000005, -8200)` | `(2800.0, -8200)` | `4.547473508864641e-13` |
| 11 | `(1800.0000000000002, -8200)` | `(1800.0, -8200)` | `2.2737367544323206e-13` |

其他 hole 也有約 `1.14e-13` 的短邊，表示不是只需要處理一個壞節點。

因果鏈為：

1. Pipeline 的 planar plan 已持有 `coordinate_tolerance=0.001`。
2. 不同 front 獨立還原世界座標：`hole_2` 的 X 為 `2800.0000000000005`，`hole_5` 為 `2800.0`。[Holed builder](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/generators/holed_rectilinear.py:248) 將後者延伸為 geometry cut，與 `hole_2` 相交時新增近重合錨點。該 generator 的容差用於不同邊界的間距檢查，**不會合併所有近重合的 mandatory coordinates**。
3. global grid 使用精確座標集合／鍵，因而把上表的近重合座標保留為不同 interface anchors。
4. [full adapter](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/adapter/_full.py:305) 將這組 global interface 轉為固定 `outline_o`；最後一層呼叫沒有傳入絕對幾何容差。
5. [source preparation](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/adapter/layer/_planning/source.py:193) 另行計算機器精度／ULP 門檻。本案 scale=`2600`，normalized tol=`2.842170943040401e-14`，換回世界座標為 `7.389644451905042e-11 μm`。[Target validation](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/adapter/layer/_planning/front.py:102) 發現沿邊長度過小，合理地拒絕該極短邊。

**處置方向：** 在 shared interface 形成以前，統一近重合 support／anchor 的 canonical identity，讓規劃、global grid、fixed layer 與 assembly 共用同一份映射。不能只放寬 `_outline_o_directions`、刪除 target 中一個 ID，或在最後做全域 weld；相鄰網格的 connectivity、固定界面與覆蓋面積必須一起保持一致。

Case1 的量化診斷及重現入口保存在 [case1 診斷說明](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/diagnosis.md)。

## 全專案盤點範圍與證據

檢視 `src/mesher` 全部 **111 個 Python 檔案、27,955 行**，包括 adapter／unit templates、grid generators、circular V1/V2、editor、quality、mesh25d、meshcontrol、mesh3d、contract、worker、export、visualization 及 `optimal_25d`。靜態索引列出 70 個檔案中的 1,618 個候選位置；**候選位置不等於 1,618 個 bug**，其中包含正確的數值 epsilon、整數拓樸判斷、註解及顯示格式。

可逐檔追查的 [source inventory](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/source-inventory.json) 保存行號、函式、原始片段與每個檔案的 SHA-256。三份附錄包含各子系統的完整 tolerance family、應維持精確的判斷及遷移注意事項：

- [Adapter、layer、unit 與 grid](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/adapter-and-grid.md)
- [Circular 與 quality](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/circular-and-quality.md)
- [Pipeline、Z、材料指派與 export](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/pipeline-and-3d.md)

`archive/` 為舊版程式；其中 `0.0001`／`0.001` 混用需維持隔離，不應回流成新實作。`optimal_25d/utils.py` 的舊式 line 判斷另有 `0.01`；本次未找到現行 pipeline 對它的引用，建議明確標示實驗性或排除發佈。`build/`、`dist/` 為生成物，修正應由原始碼重建。外部 `process_flow_kernel` 是另一個專案；此 mesher 入口只呼叫其 semantic-key validation，不能把本次結論擴張成整個上游專案已符合容差。

## 優先處理的已驗證行為

下表的 P1 表示違反本次容差需求，或已造成不穩定拓樸、錯誤／遺失幾何；P2 為其他應在遷移中處理的契約或邊界問題。已存在的精確幾何契約與新需求衝突，也列入遷移工作，但不一概稱為現有文件的實作 bug。

| 編號 | 優先 | 已驗證行為 | 主要位置／證據 |
|---|---|---|---|
| F01 | P1 | Case1 保留 `4.55e-13` 界面邊，最終建層失敗 | case1；`_full` → holed grid → `front` |
| F02 | P1 | 公開 pipeline 沒有統一 tolerance 入口；schema 也不接受該設定 | [contract](/Users/henry/Desktop/code/mesher/src/mesher/contracts/process_flow_2_5d.py:43)、附錄 pipeline 第 1 項 |
| F03 | P1 | KOZ `0.05` 將全模型 XY tolerance 從 `.001` 降成 `.0005`；KOZ `.0001` 降成 `1e-6` | [koz_tolerance](/Users/henry/Desktop/code/mesher/src/mesher/mesh25d/geometry/koz.py:23) |
| F04 | P1 | Layer 的實際距離容差只來自 epsilon；unit square 約 `2.84e-14`。`axis_tolerance=.001` 是斜率，長邊可接受 `.5` 偏移、短邊卻拒絕 `.0005` | [source](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/adapter/layer/_planning/source.py:193)、附錄 L1 |
| F05 | P1 | Holed／region／guide 路徑拒絕 `1e-10`～`.0005` 的近軸向偏差；部分 circle guide 被靜默略過 | [holed geometry](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/generators/_holed_geometry.py:101)、附錄 L3、C3、pipeline 第 7 項 |
| F06 | P1 | 兩條 pattern 相差 `1e-10`，即使 coordinate tolerance 為 `.001`，仍產生最小邊約 `1e-10` 的元素 | 附錄 L4；與 F01 同類精確錨點風險 |
| F07 | P1 | 同一網格只平移 X=0、1、10，dense seams 分別為 0、2、4 | [dense threshold](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/adapter/layer/_planning/dense.py:40)、L2 |
| F08 | P1 | V2 quad 的一個角恰為平直角時成功；該角偏移 `1e-14` 或 `1e-12` 反而失敗；cleanup 與 split 對平直角判斷不同 | [cleanup](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/utils.py:2656)、[split](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/imprint_v2/utils.py:1375)、C1 |
| F09 | P1 | Circular extension 拒絕半徑偏差 `.0005`；symmetry 圓心偏差 `.0004` 也被拒絕 | [extension](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/extend/mesher.py:126)、C2、pipeline 第 6 項 |
| F10 | P1 | Z_POINT `.5`／`.5004` 產生 `.0004` 厚 solid；相同量級原始 body 厚度卻因 rounding 靜默變成 0 個元素 | [geometry Z](/Users/henry/Desktop/code/mesher/src/mesher/mesh25d/geometry/primitives.py:51)、[plan3d](/Users/henry/Desktop/code/mesher/src/mesher/mesh25d/plan3d.py:36)、pipeline 第 3 項 |
| F11 | P1 | `.3/.1` 的 floor 誤差使 Z 只有兩層 `[0,.15,.3]`；XY 也有類似問題。簡單 grid 的 ceil 則使 `.3` 和 `.1+.2` 分成 3／4 格 | [extrusion](/Users/henry/Desktop/code/mesher/src/mesher/mesh3d/extrusion.py:503)、[rectilinear](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/generators/rectilinear.py:34)、L6／pipeline 第 4 項 |
| F12 | P1 | Material selection 預設 `.01`，納入整個位於邊界外 `.002`～`.004` 的 cell；相同 `.001` 下 BOX 與等價 POLYGON 判定不同 | [Dragger](/Users/henry/Desktop/code/mesher/src/mesher/mesh3d/extrusion.py:256)、pipeline 第 5 項 |
| F13 | P1 | 1×1 polygon 平移到 `1e7` 後被面積閾值丟棄；`1e8` 出現面積消去為 0，也改變材料密度選取 | [domain area](/Users/henry/Desktop/code/mesher/src/mesher/mesh25d/domain.py:274)、[extrusion area](/Users/henry/Desktop/code/mesher/src/mesher/mesh3d/extrusion.py:201)、pipeline 第 8 項；屬大座標風險，不是 case1 原因 |
| F14 | P1 | 簡單 grid 在原點 `1e12`、target `1e-5` 時產生 72,000 條零長度 cell edges，未先拒絕不可表示的間距 | L7；極端解析度測試，不是 case1 原因 |
| F15 | P2 | `_sort([0,.00075,.0015], .001)` 只剩 `[0]`；editor transitive equivalence 也會讓首尾相距 `.0015` 的三點全合併 | [rectilinear](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/generators/rectilinear.py:14)、[equivalence](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/utils/equivalence.py:38)、L5／core probe |
| F16 | P2 | Editor 矩形選取對 `1e-10` 邊界噪音就改變結果；沒有容差參數 | [search_element](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/editor/editor.py:56)、core probe；目前是精確 inclusive query 契約 |
| F17 | P2 | Legacy circle search 預設 `.01`；兩個 guide 差 `.0001` 在 V2 中又可能因數值級 target-agreement 判斷而凍結投影 | [circle search](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/circular/utils/topology.py:23)、C4／C5 |
| F18 | P2 | Quality checker 預設 absolute／relative 均 `1e-12`；即使 absolute=.001，其 `tau²` 面積門檻仍不能表達所有高度小於 `.001` 的扁長元素 | [quality policy](/Users/henry/Desktop/code/mesher/src/mesher/mesh2d/quality/checker.py:54)、C6 |
| F19 | P2 | Imprint 的 `minimum_area=.001` 可輸出面積 `.00005` 的 child；門檻只套用於 split 前 cleanup | C7；屬面積契約，不能與長度 `.001` 混用 |
| F20 | P2 | Z remainder 的 `min_dis=.01` 壓掉 `.005` 殘段，與 `.001` 設定脫鉤 | [meshcontrol](/Users/henry/Desktop/code/mesher/src/mesher/meshcontrol/methods.py:188)、pipeline 第 9 項 |
| F21 | P2 | CDB `.12g` 將 `1000000000.004` 寫成 `1000000000`，誤差約 `.004` | [export formatting](/Users/henry/Desktop/code/mesher/src/mesher/process_flow/exporters/cdb.py:166)、pipeline 第 10 項；須依支援座標範圍驗收 |

## 尚須隨遷移處理的靜態風險

這些位置已有程式碼證據，但本次沒有把每一條分支各自重現為失敗：

- Layer 的單一 `tol` 同時進入長度、面積、比例與節點 registry；若直接改成 `T/scale`，`h <= 32*tol` 將意外拒絕尺寸 ≤ `.032` 的元素，其他 overlap 判斷也可能容忍真實重疊。詳見 adapter 附錄的 migration hazards。
- Circle planner 的 projection=`.2*h`、guide=`max(.001*h,1)`、merge=`.002*h`、minimum area=`.0004*h²` 是操作距離／品質參數；其中會影響 identity 的用途應拆出來，不能假設等於全域幾何容差。
- Guide caches、prepared plans 與資料字典沒有統一 policy identity；不同容差的物件可能被重用。應記錄 policy／frame，不能跨 policy 無聲沿用。
- 正規化前，Shapely predicates、座標 dict/set、精確 clipping 與來源 reference keys 可能產生不同拓樸；正規化後，部分精確比較反而應保留。
- Mesh3D 儲存結構沒有完整的有限座標、正體積、Jacobian、collapsed-layer 後置驗證；目前上游錯誤可進入 export。

## 容差契約的建議定義

採用 immutable、每次 build 明確攜帶的 `GeometryPolicy`，其 `distance=0.001` 為單一來源。既有 quality 的 `GeometryTolerance` 是另一個 API，遷移時建立明確橋接，不以同名不同意義混用。

| 類型 | 建議語意 |
|---|---|
| 幾何距離 | 模型單位下的絕對距離 `T=.001`；Euclidean 點距離、點到線段距離、半徑 residual 使用一致定義；邊界包含 `≤T` |
| 數值誤差 | ULP／epsilon 與局部座標框架推導；不能自動取代或放大全域 T。若表示誤差已不容許可靠判定，回報 numeric-resolution error |
| 角度、斜率、參數 | 獨立命名；幾何距離換算參數時使用 `T / edge_length`，換算圓周角時考慮 radius；禁止把 T 直接當 radians 或 ratio |
| 面積／Jacobian／體積 | 使用正確量綱與局部尺度；近共線可依距離乘局部邊長分類，coverage 依周長、位移與運算誤差推導；保留嚴格反轉／重疊檢查 |
| 搜尋及投影距離 | band、projection reach、mesh size 是演算法控制，不等於相等判斷；用獨立欄位並限制對 identity 的影響 |
| 拓樸與 ID | node ID、連通性、parity、edge-use count 必須精確；容差用在建立 canonical identity 之前 |
| 小特徵及材料邊界 | T 不自動代表所有尺寸小於 T 的特徵可刪除；無法同時保留的 constraint／材料區間必須明確報錯並附來源 |

「相距 ≤T」不是具傳遞性的等價關係：`0, .0008, .0016` 中，相鄰均符合 T，首尾卻不符合。必須把 **contact predicate** 與 **node union** 分開，不能承諾所有輸入都能安全吸附。

建議 canonicalization 對同一語意界面的候選建立 proximity graph；若連通群直徑超過 T、材料／固定錨點相衝突，回報 ambiguity。其餘群使用穩定的來源錨點，記錄 original→canonical 映射及位移。這樣能對成功接受的輸入提供確定性與位移界限；避免依輸入順序挑第一個近鄰，也避免 `round(x,3)` 的格線邊界不連續。

`Mesh2D` 的 Z 必須精確等於 0 是目前刻意定義的資料模型不變量，不建議直接鬆綁 constructor。若輸入轉接需要接受 `|z|≤T` 的平面噪音，應提供明確 normalization，投影後建立 Z=0 的 Mesh2D。同樣地，editor cleanup 只刪除精確零面積、equivalence 為 transitive grouping，都是現有契約；遷移須明確版本化，不可暗中改變資料編輯語意。

## 分階段實施計畫

| 階段 | 交付內容 | 完成條件 |
|---|---|---|
| 1 固定問題證據 | 將 case1、極短 interface、near-flat quad、平移 dense、Z／計數等最小案例轉為正式 regression tests；保存現有 1,068 項測試基準 | 測試能精準命中錯誤 predicate／結果，且驗證失敗時輸入不變 |
| 2 共用契約與傳遞 | 新增單一 policy、單位／frame／numeric budget helpers；規劃 JSON schema 與 public API 相容遷移；worker、pipeline、plan2d/3d、adapter、circular、quality 接收同一 policy | `.001` 從入口可追到每一幾何階段；KOZ 不再靜默縮小 policy；diagnostics 顯示有效值 |
| 3 修復 canonical interface 與 case1 | 共用 supports／anchors 與來源 ID；在 global grid 生成前解決近重合座標；full planning、holed grid、fixed layer、registry／assembly 共用映射；同時處理 exact-axis 與 dense/count 邊界 | case1 全流程完成；所有界面 conforming、沒有重複座標造成的零邊／裂縫／T-junction／重疊；固定來源 identity 與幾何覆蓋正確 |
| 4 統一其他幾何階段 | Z events／controls／references 共用 canonical boundaries；移除固定小數 rounding；共用穩定 subdivision count；統一材料 membership；修復 circular cleanup→split→triangulate 一致性 | 不再出現 `.0004` 假層或無診斷材料消失；BOX/POLYGON 等價；near-flat／radius／guide 的容差邊界行為一致 |
| 5 數值與輸出完整性 | 面積與 winding 改用局部原點；不可表示尺寸 preflight；統一 final quality／volume checks；CDB roundtrip 精度；prepared cache 綁定 policy | 平移不改變有效幾何／材料；輸出不存在未診斷的 invalid cells；序列化座標誤差受保證 |
| 6 回歸與相容遷移 | 全測試、案例與性能測試；明確更新舊 exact-axis、transitive merge、minimum_area 等文件及預期結果 | 所有接受的模型符合新契約；不接受的 ambiguity 有來源、殘差與修復提示；不能只以不拋 exception 作為通過 |

階段 3 的 case1 需優先交付，但以階段 2 的最小共用 policy／canonical-interface 支援為前提。階段 4 的 Z／材料與 circular 可以在契約確定後分開實作。每次變更應保留 atomic commit／rollback，以及既有 NumPy-only core 的依賴邊界。

## 驗收矩陣

- 距離邊界：`0`、一個 ULP、`1e-14`、`1e-10`、`.0005`、`.000999`、`.001`、`nextafter(.001, ±∞)`、`.001001`；兩側擾動與正負座標。
- 變換：平移 `0/1/10/1e6`，另做 `1e7/1e8/1e12` 的支援範圍／解析度測試；測試單位轉換時也同步轉換 T，不混淆「改變幾何尺寸」與「改變表示單位」。
- 確定性：輸入、element、guide 順序重排；outline 起點、winding；連鎖近重合、競爭錨點、跨材料及細 KOZ。
- 跨階段：plan 與 build、simple／holed／full、source 與固定 target、circle cleanup／split／quality、geometry event 與 reference Z、輸出與重新解析保持一致。
- 正確性：interface 共享 ID、coverage、正 Jacobian／volume、材料標記、無裂縫／非法重疊、最大位移與 ambiguity diagnostics；禁止以刪除壞元素而留下洞洞作為成功。
- 性能：維持空間索引／分桶，避免全域 N×N 距離矩陣；以 case1 及既有大型 mesh benchmarks 比較耗時與記憶體。

不是每個靠近真實拓樸臨界點的擾動都應強迫輸出相同網格。對幾何／材料語意無法唯一判定的情況，驗收應要求穩定、具體的 conflict error；對單純浮點殘差則要求穩定結果。

## 本次驗證與重現方式

現有完整測試：`PYTHONPATH=src venv/bin/python -m unittest discover -s tests -t . -q`，**1,068 tests，112.761 秒，OK**。它證明目前測試基準通過，並不證明上述新需求已滿足。

環境版本及執行資訊保存在 [environment.json](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/environment.json)。以下是會列印目前不一致行為的診斷程式，**不是修正後的 passing regression tests**。從 repo root 執行：

```bash
PYTHONPATH=src venv/bin/python docs/tolerance-audit/core_probe.py
PYTHONPATH=src venv/bin/python docs/tolerance-audit/adapter_probe.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/mesher-tolerance-audit-mpl venv/bin/python docs/tolerance-audit/pipeline_probe.py
PYTHONPATH=src venv/bin/python docs/tolerance-audit/circular_probe.py
```

實測輸出： [core](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/core_probe.output.jsonl)、[adapter](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/adapter_probe.output.txt)、[pipeline](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/pipeline_probe.output.jsonl)、[circular](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/circular_probe.output.txt)。Case1 的獨立診斷只在執行程序內包裝函式以取得中間資料，沒有修改原始碼或輸入檔。
