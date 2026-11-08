# 絕對幾何容差遷移說明

本次修改讓幾何入口共用不可變的 `GeometryTolerance(absolute=0.001)`。
數值單位沿用模型座標，沒有相對比例或依網格尺寸自動縮放。每次公開操作解析
一份設定，子階段保留同一份設定。模型換單位時，座標、操作距離及 absolute
必須同比例換算。

```python
from mesher import GeometryTolerance
from mesher.mesh2d import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh

policy = GeometryTolerance(absolute=0.001)
mesh = generate_rectilinear_mesh(0.1, [0, 0.3], [0, 0.3], tolerance=policy)
indices = EditorMesh2D.search_element(mesh, 0, 0, 0.3, 0.3, tolerance=policy)
```

## API 變更

| 舊介面 | 新介面／行為 |
| --- | --- |
| grid/full adapter `coordinate_tolerance` | `tolerance`，預設絕對距離 0.001 |
| layer adapter `axis_tolerance` | `tolerance`，近軸向以實際偏離距離判斷，與邊長無關 |
| circular V2 `projection_tolerance` | `projection_distance`，仍是主動投影範圍，可為 0 |
| circular V2 `guide_tolerance`、`merge_tolerance` | 共用 `tolerance`；guide、圓周、合併及 split 使用一致幾何距離 |
| editor `equivalence(mesh, tol)` 或 `tol=...` | `equivalence(mesh, tolerance=policy)`，keyword-only |
| quality 原有 `GeometryTolerance(relative=...)` | 改用核心 `GeometryTolerance(absolute=...)`；移除相對幾何容差 |
| quality、editor 查詢、Dragger | 接收共用 `tolerance`；BOX/POLYGON/CIRCLE 的邊界包含一致 |
| `Pattern2D.coordinate_tolerance/clearance_tolerance` | `Pattern2D.tolerance`；KOZ 不再縮小全模型設定 |
| 幾何容差 0 | 不再接受；明確提供正容差。需要保留固定座標時使用固定約束，而非零容差分支 |
| 帶 Z 噪音的平面輸入 | 明確呼叫 `Mesh2D.from_planar(..., tolerance=policy)`；原建構式仍要求 Z 精確為 0 |

舊關鍵字移除，錯誤呼叫會直接失敗，避免新舊規則混用。角度、品質分數、面積、
Jacobian、體積、網格尺寸與擴展距離保留各自單位；不能把這些參數全部設成 0.001。
局部數值誤差另算，不會自動放寬或縮小 T。

## Process-flow 文件

新文件可使用：

```json
{
  "schemaVersion": "1.1.0",
  "unitSystem": "um",
  "mesher": "process_flow_2_5d",
  "globalElementSize": 1000,
  "geometryTolerance": 0.001,
  "symmetry": "full",
  "controls": []
}
```

`geometryTolerance` 必須有限且嚴格大於 0。既有 `1.0.0` 文件仍可直接讀取，
未提供設定時採用 0.001；case1 的原始 JSON 不需修改。Python `build_mesh` 與 CLI
都從同一個 mesh-control 文件解析，prepared geometry、2D plan 與 Z plan 傳遞此設定。
進度診斷及 CDB metadata 包含有效容差。Adapter 的 prepared source、plan 與 result
保留相同 policy identity，diagnostics 分開記錄幾何容差及 numerical error budget。
CDB 座標採 float64 roundtrip 精度。

## 幾何與失敗契約

- 距離恰好等於 T 屬於接觸／近重合。臨界點距離使用精確 fallback，不以固定
  額外 epsilon 放寬 T。
- 相容支撐線在 visibility/intersection 前共同正規化。每個點的總歐氏位移不得
  超過 T。來源網格和固定節點座標、ID 保持精確；最後不靠全域 weld 修補裂縫。
- 近鄰連通群的完整直徑不得超過 T。固定座標優先，沒有固定約束時採穩定座標排序。
  不允許 A 近 B、B 近 C 就把超過 T 的 A/C 合併。
- 非零材料層、孔洞與必要間隙無法保留時報錯。Z 幾何、控制點與材料事件共同映射，
  移除十進位 round，維持 END/START 優先序。
- Circular 在 split 前修復近平直角；`minimum_area` 是明確的最終輸出條件。
  失敗時原 mesh 保持不變。會折疊原本有效元素的投影會撤回；不能靠刪除元素留下孔洞。
- Process-flow 過去內部的 `0.0004 * element_size**2` 是 cleanup 啟發式，沒有出現在
  輸入 schema。此啟發式移除，pipeline 使用 `minimum_area=0.0` 並執行幾何退化驗證；
  直接呼叫 circular API 時明確指定的 `minimum_area` 仍嚴格驗證。
- 明確呼叫 editor equivalence 可重新編號節點；不再默默刪除失效元素。顯式 cleanup
  工具保留其清理用途，與一般 imprint 成功契約分開。

`GeometryToleranceError` 提供 `code` 與 `context`，包含階段、來源及可取得的實測值；
adapter 保留 `QuadLayerError` 介面及其診斷內容。主要精度衝突代碼：

| code | 意義 |
| --- | --- |
| `ambiguous_geometry` | 近重合鏈超過 T，或提議的正規化超過位移預算 |
| `feature_below_resolution` | 正規化或網格建構將消除必要特徵 |
| `fixed_constraint_conflict` | 不同固定座標無法共用幾何身分 |
| `numeric_resolution_exceeded` | float64 座標解析度不足以可靠執行指定 T |

非法輸入、退化元素與不滿足最小面積另有具體錯誤，不能當成自動調整容差的理由。

## 回歸測試的遷移理由

1. 依賴傳遞閉包合併、零容差、刪除薄元素的測試改為驗證結構化失敗與輸入不變。
2. 微小或縮放幾何測試使用與模型單位相符的明確正容差，不再依賴隱藏 numerical floor。
3. 以相對斜率接受長邊偏移的測試改為絕對距離；超過 T 的偏移應被拒絕。
4. 圓形案例將投影距離與共用容差分開；部分過去靠大範圍 weld 或刪除格子通過的
   案例改為驗證保守投影及無孔洞結果。guide 正規化後仍檢查總位移預算。
5. 浮點細差引起的超短邊、Z 微層、floor/ceil 多一格改為明確的拓樸、數量及體積 assertions。
6. CDB 測試從固定 12 位數字改為逐值 roundtrip；數值文字格式可能改變（例如 `1` 變為 `1.0`），讀回後的 float64 座標保持一致。

正式回歸位於 `tests/unit/test_tolerance.py`、`tests/unit/mesh2d/test_tolerance_contract.py`、
各子系統 tolerance 測試，以及 `tests/integration/mesh25d/test_case1_tolerance.py`。
case1 測試保留原始 1.0.0 文件，驗證 8 個界面、來源座標/ID、外框覆蓋、材料範圍、
解析體積及 CDB roundtrip，不只檢查沒有 exception。

原 `docs/tolerance-audit/` 是修改前的調查快照；其程式片段、輸出與行號記錄當時 API，
不是本版本的執行範例。當前行為以正式測試與本遷移說明為準。

## 保留的數值門檻

原始碼中的 machine epsilon 並非第二套幾何容差：

- 點、圓與線段的臨界距離使用快速浮點判斷，接近 T 時以有理數精確比較儲存的
  float64 輸入，不透過增加 T 吸收誤差。
- unit builders 的 64/128 倍 machine epsilon 用於局部面積、Jacobian、方向與
  座標轉換誤差；full adapter 的 roundtrip 檢查同樣屬數值檢查。
- `1e-10`／`1e-11` 的相對門檻保留於面積加總守恆驗證，量綱是面積。
- subdivision 的近整數判斷使用 32 倍 machine epsilon；不允許以 T 除以網格尺寸
  放寬 floor/ceil 計數。角度和操作距離仍由對應參數決定。

## 本次驗收結果（2026-11-07）

完整 discovery 共 **1,148 項測試通過**，0 failure、0 error、0 skip，耗時 155.474 秒。

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/mesher-tolerance-audit-mpl \
  venv/bin/python -m unittest discover -s tests -t . -q
```

原始 `/Users/henry/Downloads/case1` 的兩份 JSON 與正式 fixture 逐 byte 相同。
未修改 schema 1.0.0 文件或加入容差 override 即完成 2D、3D 與 CDB export。

| 驗收項目 | 結果 |
| --- | --- |
| 保留來源網格 | 77,275 nodes、77,197 elements，座標、ID 與 connectivity 完全一致 |
| Region interfaces | 8 個；節點數 18、18、20、18、16、24、16、16；精確共用 incidence |
| 2D 覆蓋面積 | 1,600,000,000 平方模型單位 |
| 2D 最短邊 | 約 3.125，沒有浮點殘差短邊 |
| 最終 3D 網格 | 322,645 nodes、234,919 elements |
| 材料 Si 體積 | 1,240,000,000,000 立方模型單位 |
| 材料 si 體積 | 79,000,000,000 立方模型單位 |
| 總體積 | 1,319,000,000,000，符合解析值 |
| CDB | 全部節點讀回後與原 float64 座標逐值一致 |

另以獨立幾何運算重跑 20 個 circular 歷史案例，缺失面積與封閉孔洞均為 0。
面積加總存在浮點殘差，最大測得重疊面積約 `0.000732422`，低於該大尺寸案例的
數值測量誤差預算；此測量值沒有用來改變幾何容差。

效能使用既有 benchmark 驗證，fixture 建構不計入時間：

| 測試 | 實測 | 既有時間門檻 |
| --- | --- | --- |
| 完整單層 adapter，5 次中位數 | 0.253 秒 | 1 秒 |
| Editor 10 萬元素相關 equivalence 案例，各 3 次中位數 | 0.279–2.447 秒 | 每案例 10 秒 |
| 一般 LINE / dense LINE | 0.0096 / 0.0059 秒 | 30 秒 timeout |
| 800 holes，單次抽測 | 0.301 秒 | 保留空間索引；`validation_pairs=0` |

`git diff --check` 通過。舊 API 測試的語義變更理由列於上方；沒有以跳過測試或
放寬幾何 assertions 取代修正。
