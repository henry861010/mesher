# `generate_layer_adaptive_mesh` 使用指南

`generate_layer_adaptive_mesh()` 在既有 Quad4 網格的封閉邊界上建立一層轉接網格。
它可以自動向外擴展，也可以接到指定的 `outline_o`。若另一個區塊已經完成網格化，
使用 `fixed_outer_boundary=True` 保留接合邊界的所有節點與邊段。

函式成功時會更新傳入的 `mesh`，並回傳 `QuadLayerResult`；任何失敗都不改動原始 mesh。
新增元素皆為非退化、凸、逆時針的 Quad4，所有節點的 Z 必須為 0。

## 1. 最小範例：自動建立一層

在專案根目錄執行範例時，使用 `PYTHONPATH=src venv/bin/python`。

```python
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh

mesh = generate_rectilinear_mesh(1.0, [0, 4], [0, 4])
outline_i = EditorMesh2D.outlines(mesh)[0]

result = generate_layer_adaptive_mesh(
    mesh, outline_i, element_size=1.0,
    expand_size=2.0,
)
assert result.mesh is mesh
# result.outline 是完整的 node-ID loop，可以作為下一次的 outline_i。
assert result.diagnostics['added_elements'] > 0
```

這裡 `element_size=1.0` 控制目標網格尺寸，`expand_size=2.0` 控制擴展距離。
沒有平行 pattern 介入時，各邊擴展 2.0。兩個尺寸可以分開設定。

## 2. 先選使用模式

| 使用目的 | `outline_o` | `fixed_outer_boundary` | 外框節點行為 |
| --- | --- | --- | --- |
| 自動擴展一層 | `None` | `False`，預設 | 自動產生外框並取樣 |
| 指定外框形狀與必留節點 | 完整 node-ID loop | `False` | 保留提供的節點，允許在其間補點 |
| 連接已完成網格的另一區塊 | 完整 node-ID loop | `True` | 保留所有 ID、精確座標及相鄰邊段，禁止補點或刪點 |

固定的是 **`outline_o` 邊界**。轉接區內部與內部接線仍可新增節點。
`fixed_outer_boundary=True` 搭配 `outline_o=None` 會報錯。

### 輸入邊界的契約

- `mesh` 必須至少已有一個合法 Quad4。原始元素與 node ID 不會重新編號、切割或移動。
- `outline_i` 與 `outline_o` 都是一維整數 node-ID loop，索引同一個 `mesh.nodes`。
  不接受 XY 座標陣列作為 `outline_o`；新座標須先加入 `mesh.nodes`，再傳入 ID。
- loop 隱含閉合；最後可選擇再放一次第一顆 ID。起點可以在直邊中間。
  `outline_i` 的外輪廓必須順時針、孔洞必須逆時針；`outline_o` 仍接受兩種繞行方向。
- `outline_i` 必須包含每一顆 source 邊界節點，相鄰 ID 必須是可接合的既有邊界邊段。
  此為呼叫者責任，API 不透過既有元素 ownership 驗證 source；違反前置條件時，
  不保證提前拒絕或特定錯誤碼。
  邏輯邊需接近全域 X／Y 軸，相鄰邏輯邊互相垂直。
- `outline_o` 必須是簡單、水平／垂直的封閉 loop。忽略共線中間點後，其轉向順序須對應
  **repair 之後的 source**，每條對應邊都必須位於擴展側，且與 source 留有正間距。
- 演算法保留 source 輸入方向，不自動反轉，固定向左側建立新網格。
  呼叫者須讓既有網格位於右側；順時針外輪廓向外擴展，逆時針孔洞向孔洞內延伸。
  `EditorMesh2D.outlines(mesh)` 可提供符合此方向契約的邊界。
- 本 API 建立的是兩條對應封閉邊界之間的完整帶狀區域。兩個任意並排區塊的開放邊界接合不屬於此介面。

固定模式的 `result.outline` 可能調整起點與方向，但代表相同的 ID 循環與相鄰邊段。
接到既有元素時，必須提供完整的接合邊界；已有兩個元素使用的邊不能再加入第三個元素。

## 3. 完整參數表

所有長度使用 `mesh.nodes` 的座標單位；沒有內建毫米或公尺假設。
除前三個位置參數外，其餘皆為 keyword-only。

| 參數 | 預設 | 合法值／單位 | 用途與互動 |
| --- | --- | --- | --- |
| `mesh` | 必填 | `Mesh2D`，Z=0 | 成功時原地更新；失敗保持原狀 |
| `outline_i` | 必填 | 一維整數 node-ID loop | 完整 source 邊界 |
| `element_size` | 必填 | 有限正長度 | 網格目標尺寸；未指定 `expand_size` 時也作為擴展基準 |
| `pattern_lines` | `()` | 有限 XY 線段，形狀 `(n, 2, 2)` | 擴展參考及必須保留的邊／接線限制 |
| `outline_o` | `None` | 一維整數 node-ID loop 或 `None` | 指定完整目標外框；固定模式必填 |
| `fixed_outer_boundary` | `False` | Python／NumPy boolean | `True` 時禁止外框新增、刪除或移動節點 |
| `expand_size` | `None` | 有限正長度或 `None` | 控制擴展厚度；`None` 取 `element_size` |
| `expand_ratio` | `1.0` | 有限數值，≥1 | 搜尋平行 pattern 的距離範圍倍率 |
| `axis_tolerance` | `1e-3` | 有限數值，`0 ≤ value < 1` | source 邏輯邊相對長度的橫向偏差容許值；不移動 source |
| `max_added_elements` | `1_000_000` | 正的 Python／NumPy 整數 | 本次 repair 與轉接新增 Quad4 的共同上限；等於上限可接受 |
| `max_transition_count` | `20` | 正的 Python／NumPy 整數 | 一般 LINE 外框一段最多對應幾個 source 邊段；固定模式停用 |
| `min_size_ratio` | `0.5` | 有限數值，`0 < value ≤ 1` | 一般 LINE 外框取樣間距下限相對 `element_size` 的比例；固定模式停用 |
| `repair_slots` | `True` | Python／NumPy boolean | 先修補受擴展覆蓋的支援槽口與孤立階梯轉角 |
| `repair_max_element_size` | `None` | 有限正長度或 `None` | repair 內部目標尺寸上限；不是所有邊長的硬上限 |
| `small_element_size` | `None` | 有限正長度或 `None` | 辨識短邊密集段的門檻；`None` 關閉額外密集段切分 |
| `dense_mapping_angle` | `30.0` | 有限度數，`0 ≤ value < 90` | 密集段端點的單側映射角度；不限制轉角選點 |

尺寸、比例、角度與計數參數不接受 boolean；計數不接受浮點數或 `None`。
極小尺寸若低於輸入座標的可表示精度，也會被拒絕。

### 三種 size 的差異

- `element_size`：希望生成的網格大約多大。
- `expand_size`：新的一層要延伸多遠。
- `small_element_size`：哪些 source 邊段應視為密集區，進而嘗試在端點切分。

`small_element_size` **不是最小元素尺寸**，也不保證密集區內每個新元素都小於該值。
固定模式保留原有外框間距，因此 `element_size` 也不會強迫外框補點。

## 4. 參數優先順序

| 規則 | 一般模式 | 固定外框模式 |
| --- | --- | --- |
| 自動外框 | 未提供 `outline_o` 時，依擴展基準與 pattern 決定 | 不適用，必須提供外框 |
| 已提供 `outline_o` | 使用指定外框，pattern 不會吸附或移動它 | 相同，並鎖定完整節點與邊段 |
| `max_transition_count`／`min_size_ratio` | 約束重採樣的 LINE 外框 | 數值仍驗證，但不施加這兩項限制 |
| 密集段接點不存在 | 可建立合法法線投影點 | 只略過該端的額外分界 |
| 必要 pattern 接點不存在 | 可依一般規則建立新外框接點 | 報錯，不以附近 node 取代 |
| 元素預算、幾何、拓撲及 Quad4 驗證 | 全部適用 | 全部適用 |

自動擴展時，基準長度為 `expand_size`，若為 `None` 則取 `element_size`。
在有投影重疊的平行 pattern 中，最近正距離 `d` 若滿足
`d ≤ 基準長度 × expand_ratio`，整條邏輯邊擴展到該 pattern；否則使用基準長度。

**repair 先於指定外框套用。** 即使有 `outline_o`，repair 的覆蓋判定仍使用擴展基準、
`expand_ratio` 與 pattern。repair 完成後才以 `outline_o` 建立轉接；不相容時直接失敗，
不會因指定外框再重做一次 repair。

一般 LINE 的外框取樣間距位於
`[min_size_ratio × element_size, element_size]`。
已存在的相鄰必要接點若比下限更近，保留為一段並記錄例外，轉接數量上限仍適用。
這些規則不限制 corner、repair 內部或每一條內部元素邊的長度。

## 5. 密集段：單側角度與第一顆 node

同一 LINE 中，連續且長度 **嚴格小於** `small_element_size` 的 source 邊段組成密集段。
單一短邊也符合；整條 LINE 都是密集段時，已有兩端分界，不需要再切。
corner 與 pattern 已確立的分界優先保留。

以正規化 LINE 的行進方向作為正方向，法線朝向外框為 0°：

```text
                       outline_o
       begin 可選區                   end 可選區
           ●   ●                         ●   ●
             \ |                         | /
              \|                         |/
source ───── begin ===== 密集段 ======== end ─────→ 行進方向
             -angle～0°                 0°～+angle
```

- begin 從 0° 朝負方向搜尋，禁止朝密集段內部偏移。
- end 從 0° 朝正方向搜尋，同樣禁止朝密集段內部偏移。
- 每端只選第一顆合法 node。例如 begin 有 −5°、−20° 時選 −5°。
- 角度上限包含邊界；`dense_mapping_angle=0` 只接受法線方向上的重合接點。
- 搜尋限於當前 corner／pattern 分界之間，且不能讓分界逆序、交叉、離開帶狀區或形成零寬 LINE。
- 一般模式可使用當前已確立的外框接點；沒有合法候選時，嘗試新增法線投影接點。
  固定模式只使用 `outline_o` 的既有 node。新的 dense 投影不會變成其他端點的候選。

### 只找到一端仍然切分

若 begin 找得到、end 找不到，固定模式產生：

```text
原 LINE：pre ─── begin ===== dense ===== end ─── remaining
切分後 ：[pre → begin] [begin → remaining]
```

找到 end 時同理。兩端都找不到則維持原 LINE。
這仍能約束找到的一側；另一側的斜率與元素品質取決於既有外框及後續 Quad4 配對。
`dense_mappings` 會逐端記錄採用或略過的原因。

## 6. 轉角：不限角度的最近合法點

轉角選點不使用 `dense_mapping_angle`。

```text
A ───── S ── source 邊
│       │
C ───── B ── outline_o
```

外凸轉角的 `S` 是 source 轉角，`C` 是對應外框轉角；A、B 是轉角區與 LINE 的接點。
一般模式沿用法線投影。固定模式在對應的兩條外框邊上，各自尋找距離 S 最近的既有 node。
若最近點是 C，排除 C 並沿該側邊鏈尋找下一顆，避免轉角區退化。

內凹轉角則保留原來的對應做法：在 source 兩側找最接近外框轉角的 node，
排除 source 轉角本身，使兩側都保留有效邊段。

候選按距離排序，同距採正規化邊界順序。若組合造成交叉、重疊、逆序或退化，
會嘗試其他候選並回退衝突組合。pattern 及 repair 已確立的必要共用接線不會任意移動。
此搜尋改善分區成功率；生成仍需通過 Quad4 單元與共用接線配對，不能保證任意幾何皆可成功。

## 7. 指定外框，並先預覽

```python
import numpy as np
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh, plan_quad_sections
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh

mesh = generate_rectilinear_mesh(1.0, [0, 4], [0, 4])
outline_i = EditorMesh2D.outlines(mesh)[0]
xy = np.array([[-2., -2.], [-2., 6.], [6., 6.], [6., -2.]])
outline_o = np.arange(mesh.node_count, mesh.node_count + len(xy), dtype=np.int64)
mesh.nodes = np.vstack((mesh.nodes, np.column_stack((xy, np.zeros(len(xy))))))

saved_nodes, saved_elements = mesh.nodes.copy(), mesh.elements.copy()
plan = plan_quad_sections(mesh, outline_i, 2.0, outline_o=outline_o)
assert np.array_equal(mesh.nodes, saved_nodes)
assert np.array_equal(mesh.elements, saved_elements)

result = generate_layer_adaptive_mesh(mesh, outline_i, 2.0, outline_o=outline_o)
assert set(outline_o).issubset(result.outline)
```

這個範例允許增加外框節點。只給矩形四角通常不足以在固定模式下形成合法轉角分區。

`plan_quad_sections()` 接受相同的幾何、repair、固定外框與密集段參數；不接受
`max_added_elements`、`max_transition_count`、`min_size_ratio`。
它預覽分區、共用接線、外框及實際擴展距離，不保證後續建構能符合預算或找到 Quad4 配對。
固定模式的預覽另會檢查外框接點與邊界節點數的奇偶相容性。

若需要 repair，預覽會在暫存資料上修補，`plan.prepared_source` 提供唯讀 nodes、elements 與 outline。
此時 section 的 source ID 可能索引該暫存快照，而非只有原始 mesh。

## 8. 完整範例：接合兩個已完成網格的區塊

以下先建立內部方塊與外圍有孔區塊，再合併到同一個 `Mesh2D`。
合併時第二個區塊的元素與邊界 ID 必須一起加上節點位移量。

```python
from collections import Counter
import numpy as np
from mesher import Mesh2D
from mesher.mesh2d.adapter import generate_layer_adaptive_mesh
from mesher.mesh2d.editor import EditorMesh2D
from mesher.mesh2d.generators import generate_rectilinear_mesh, generate_rectilinear_mesh_optimal

inner = generate_rectilinear_mesh(1.0, [0, 4], [0, 4])
outline_i = EditorMesh2D.outlines(inner)[0]
outer = generate_rectilinear_mesh_optimal(
    2.0,
    [[-4, -4], [8, -4], [8, 8], [-4, 8]],
    holes=[[[-2, -2], [6, -2], [6, 6], [-2, 6]]],
)

offset = inner.node_count
mesh = Mesh2D(
    np.vstack((inner.nodes, outer.mesh.nodes)),
    np.vstack((inner.elements, outer.mesh.elements + offset)),
)
outline_o = np.asarray(outer.hole_outlines[0]) + offset
before_nodes, before_elements = mesh.nodes.copy(), mesh.elements.copy()

result = generate_layer_adaptive_mesh(
    mesh, outline_i, element_size=2.0,
    outline_o=outline_o,
    fixed_outer_boundary=True,
    # 如需密集段分界，可設定 small_element_size 與 dense_mapping_angle。
)

assert len(result.outline) == len(outline_o)
assert set(result.outline) == set(outline_o)
assert np.array_equal(mesh.nodes[:len(before_nodes)], before_nodes)
assert np.array_equal(mesh.elements[:len(before_elements)], before_elements)
incidence = Counter(
    tuple(sorted((int(a), int(b))))
    for cell in mesh.elements
    for a, b in zip(cell, np.roll(cell, -1))
)
for loop in (outline_i, outline_o):
    assert all(incidence[tuple(sorted((int(a), int(b))))] == 2
               for a, b in zip(loop, np.roll(loop, -1)))
```

兩條接合邊界的每個邊段最後都由兩個元素共享；原始兩區塊的座標與元素保持不變。

## 9. Repair、pattern、奇偶數與元素預算

**Repair。** `repair_slots=True` 會先填補支援的槽口與覆蓋階梯轉角。
repair 邊界取樣以既有邊段長度中位數為目標，並以 `element_size` 為上限。
預設內部目標逐層加倍至 `min(element_size, 4 × boundary_target)`。
指定 `repair_max_element_size` 時，不能小於該 repair 的 boundary target，並仍受 `element_size` 截限。
這不是每條元素邊的硬上限。若要保留嚴格單層流程，可設 `repair_slots=False`。

**Pattern。** 支援沿邊界的有限線段，或從既有 source node 到對應外框的完整接線。
不支援內部懸空端點、內部 T／X 交會、封閉 pattern loop 或切過凍結 source 邊段。
固定模式要求所有必要的外框接點已存在於 `outline_o`；不能用附近點替代 pattern 的幾何位置。
每段 pattern 的長度及端點距 source 包圍盒的 L∞ 距離，不得超過 source 最大 XY 跨距的一百萬倍。

**奇偶數。** 固定外框與修補後 source 的邊界段數必須同奇偶；可選的重複閉合 ID 不計入段數。
這是必要條件，仍需滿足局部 section／seam 的 Quad4 配對。
固定模式不會為修正奇偶數而偷偷在外框補點。

**預算。** repair 與後續轉接共同使用 `max_added_elements`，只計本次新增元素。
候選試算不累加消耗預算。等於上限可接受，每次呼叫各自計算；不保證執行時間或記憶體用量。
已選定結果超過上限時，報錯且不提交，不會為符合預算再搜尋較小網格。

## 10. 回傳值與診斷

`result.mesh` 是原本傳入的物件；`result.outline` 是新外框的平坦 ID loop。
`result.diagnostics` 常用欄位如下：

| 欄位 | 用途 |
| --- | --- |
| `fixed_outer_boundary` | 本次是否鎖定外框 |
| `line_policy_active` | LINE 重採樣的尺寸／轉接數限制是否生效；固定模式為 `False` |
| `max_transition_count`、`min_size_ratio` | 傳入並驗證後的值；是否有效須搭配 `line_policy_active` |
| `small_element_size`、`dense_mapping_angle` | 密集段門檻與角度設定 |
| `dense_sections`、`dense_seams` | 成功形成的完整密集分區與新增分界；單端切分可能留下混合 LINE |
| `dense_mappings` | 每個額外端點的 `endpoint`、`status`、`reason`；採用時另含 target ID／座標、角度及 seam ID |
| `corner_mappings` | 固定模式自由轉角接點的來源／目標 ID、替代狀態與候選順位 |
| `added_nodes`、`added_elements` | 本次新增總量，包含 repair |
| `repair_added_elements`、`expand_added_elements` | repair 與後續轉接各自新增量 |
| `edge_expansion`、`snapped_patterns` | 實際厚度與自動擴展採用的 pattern；指定外框時後者皆為 `None` |
| `quality` | 新元素最差長寬比、最小 scaled Jacobian 等品質資訊 |

`dense_mappings.status` 為 `mapped` 或 `skipped`。
`reason` 包含 `existing_contact`、`projected_contact`、`canonical_contact`、
`no_eligible_node`、`no_valid_projection`、`not_planned`。
尚未生成的自動投影接點沒有既有 ID，`target_node` 為 `None`。

完整網格 API `generate_full_adaptive_mesh()` 也接受 `small_element_size` 與 `dense_mapping_angle`。
它先規劃最後一層的接點，再交給全域網格生成器；固定接合時重用保存的接點，
即使全域網格新增了更近 node 也不重新選點。

## 11. 常見錯誤與處理

捕捉 `QuadLayerError` 後，查看 `error.code` 與 `error.context`；context 通常包含
`stage`、`parameter`、來源邊、轉角、section 或 pattern 資訊。原 mesh 會保持呼叫前狀態。

| 情況 | 處理方式 |
| --- | --- |
| 固定模式未給外框／參數無效 | 提供完整 ID loop，並依參數表修正型別、範圍 |
| 必要 pattern 外框接點缺失 | 在建立另一區塊時預先納入該接點；固定接合不能修改已完成的邊 |
| source 與固定外框奇偶數不合 | 調整上游網格的邊界分段；僅增加內部節點不能修正邊界奇偶數 |
| 無法找到合法轉角分區 | 檢查外框對應邊是否有可用的非轉角 node、間距及附近分區 |
| `unmeshable_section_layout`／`unmeshable_seam_assignment`／`unmeshable_direct_quad_layout` | 依 context 檢查分區及局部 Quad4 配對；合法預覽不代表一定可建構 |
| `line_constraints_infeasible` | 一般模式的尺寸帶與轉接數上限衝突；調整 `element_size`、`min_size_ratio` 或 `max_transition_count` |
| `element_limit_exceeded` | 檢查 `limit`、`required_elements`，調整預算或上游尺寸設定 |
| `topology_change` | 外框穿過 source、接到已佔用區域，或形成懸掛節點／非流形接合 |
| `numeric_range_exceeded` | 檢查 pattern 範圍與座標尺度 |

演算法不會透過縮小整層厚度、插入三角形或退化元素來掩蓋失敗。

## 12. 舊版參數遷移

`dense_region_tolerance` 已移除，不保留相容別名；繼續傳入會得到 `TypeError`。
請改用 `dense_mapping_angle=30.0`，單位為度，縮放模型時不需隨長度縮放。
舊版距離容差不能直接當作新角度值使用。

新規則還限制 begin／end 的偏移方向，因此原本距離很近但朝密集段內側的 node 不再被選中。
`small_element_size` 名稱保持不變。未指定 `fixed_outer_boundary` 時仍允許外框補點。
