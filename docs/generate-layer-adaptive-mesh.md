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
  建立網格時會在寫回前確認每段都恰好屬於一個既有元素。幾何預覽
  `plan_quad_sections()` 則假設傳入的邊段可供接合。
  邏輯邊需接近全域 X／Y 軸，相鄰邏輯邊互相垂直。
- `outline_o` 必須是簡單、水平／垂直的封閉 loop。每條邏輯邊的全部節點在橫向的座標跨度須在 `tolerance` 以內，讓回傳外框可以接續建層。忽略共線中間點後，其轉向順序須對應
  **repair 之後的 source**，每條對應邊都必須位於擴展側，且與 source 留有正間距。
- 演算法保留 source 輸入方向，不自動反轉，固定向左側建立新網格。
  呼叫者須讓既有網格位於右側；順時針外輪廓向外擴展，逆時針孔洞向孔洞內延伸。
  `EditorMesh2D.outlines(mesh)` 可提供符合此方向契約的邊界。
- 本 API 建立的是兩條對應封閉邊界之間的完整帶狀區域。兩個任意並排區塊的開放邊界接合不屬於此介面。

固定模式的 `result.outline` 可能調整起點與方向，但代表相同的 ID 循環與相鄰邊段。
接到既有元素時，必須提供完整的接合邊界；已有兩個元素使用的邊不能再加入第三個元素。
若新舊區域只在一個外框頂點共用節點，會意外共用該節點的 FEM 自由度。建層會以
`topology_change` 拒絕這種接合，錯誤資訊的 `node_id` 會指出接觸位置。

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
| `tolerance` | `GeometryTolerance(0.001)` | 有限正絕對距離或共用容差物件 | source 邏輯邊對軸向的實際偏離距離；與邊長無關、不移動 source |
| `max_added_elements` | `1_000_000` | 正的 Python／NumPy 整數 | 本次 repair 與轉接新增 Quad4 的共同上限；等於上限可接受 |
| `max_transition_count` | `20` | 正的 Python／NumPy 整數 | 可重採樣 LINE 的每條輸出邊最多承接幾條輸入邊，密集區的兩個 sublayer 各自套用 |
| `min_size_ratio` | `None` | `None` 或有限數值 `0 < value ≤ 1` | 一般及混合 LINE 的外框間距下限為此比例乘以 `element_size`；`None` 表示不設下限 |
| `max_size_ratio` | `None` | `None` 或有限數值 `value ≥ 1` | 一般及混合 LINE 的外框間距上限為此比例乘以 `element_size`；`None` 表示不設上限 |
| `repair_slots` | `True` | Python／NumPy boolean | 先修補受擴展覆蓋的支援槽口與孤立階梯轉角 |
| `repair_max_element_size` | `None` | 有限正長度或 `None` | repair 內部目標尺寸上限；不是所有邊長的硬上限 |
| `small_element_size` | `None` | 有限正長度或 `None` | 辨識短邊密集段的門檻；`None` 關閉額外密集段切分 |
| `small_element_group_count` | `0` | 非負的 Python／NumPy 整數 | 同一 LINE 上的連續短邊數必須超過此值才算密集段 |
| `dense_mapping_angle` | `30.0` | 有限度數，`0 ≤ value < 90` | 密集段端點搜尋既有 node 與向外展寬的單側角度 |
| `dense_min_gap_ratio` | `0.5` | 有限比例，`0 ≤ value ≤ 1` | 同一 LINE 上相鄰密集段之間需保留的最小寬度比例 |

尺寸、比例、角度與計數參數不接受 boolean；計數不接受浮點數或 `None`。
極小尺寸若低於輸入座標的可表示精度，也會被拒絕。

重跑以版本 `3d5417c` 預設值交付的模型時，可顯式傳入 `min_size_ratio=0.5`、
`max_size_ratio=1.25` 以沿用當時的 LINE 間距政策。目前兩者都預設為 `None`。
即使幾何與 `element_size` 相同，這項差異也可能改變節點、元素連接以及求解結果。

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
| `max_transition_count` | 約束所有重採樣的 LINE 外框 | 驗證數值，採用指定外框 |
| `min_size_ratio`／`max_size_ratio` | 約束一般及混合 LINE 的外框間距 | 驗證數值，採用指定外框 |
| 密集段接點不存在 | 可建立合法法線投影點 | 只略過該端的額外分界 |
| 必要 pattern 接點不存在 | 可依一般規則建立新外框接點 | 報錯，不以附近 node 取代 |
| 元素預算、幾何、拓撲及 Quad4 驗證 | 全部適用 | 全部適用 |

自動擴展時，基準長度為 `expand_size`，若為 `None` 則取 `element_size`。
在有投影重疊的平行 pattern 中，最近正距離 `d` 若滿足
`d ≤ 基準長度 × expand_ratio`，整條邏輯邊擴展到該 pattern；否則使用基準長度。

**repair 先於指定外框套用。** 即使有 `outline_o`，repair 的覆蓋判定仍使用擴展基準、
`expand_ratio` 與 pattern。repair 完成後才以 `outline_o` 建立轉接；不相容時直接失敗，
不會因指定外框再重做一次 repair。

一般及混合 LINE 優先參照內框接點的法向投影來決定外框節點。`element_size` 是目標間距。
上下限可獨立設定，例如只給 `max_size_ratio=1` 時，外框間距至多為 `element_size`，沒有比例下限。
相鄰 unit 之間只用兩點接線。頭尾接線有三點或更多必要節點時，端部 unit 會容納這些點。
粗細轉接仍可能在單一 unit 內部產生節點。
已存在的相鄰必要接點若比下限更近，保留為一段並記錄例外，轉接數量上限仍適用。
這些規則不限制 corner、repair 內部或每一條內部元素邊的長度。

選定 LINE 與 corner 的 unit 後，若最差元素的長短邊比或最小 scaled Jacobian 的倒數大於 4，
adapter 會對接觸這些元素的新增內點做一輪局部平滑。每個點朝直接鄰點的平均位置先走半步，
必要時再試四分之一步。相鄰 Quad4 都須通過凸性與邊長檢查，且整圈的最差邊長比和
最小 scaled Jacobian 至少一項改善、另一項持平或改善。unit 邊界與共用接線維持原座標。
尖角案例的座標推算、品質指標與重現方式見 [Quad4 unit 的局部平滑](quad4-unit-smoothing.md)。

## 5. 密集段：單側角度與第一顆 node

每條 LINE 分別計算長度 **嚴格小於** `small_element_size` 的連續 source 邊數。
遇到長度大於或等於門檻的邊，或遇到 corner、pattern 分界，就重新計數。
連續短邊數必須 **超過** `small_element_group_count` 才算密集段。例如設為 `2` 時，
至少要連續 3 條短邊。預設為 `0`，單一短邊即可構成密集段。整條 LINE 都符合時沿用既有兩端分界。

重採樣時，整個 LINE 都落在密集段內，外框間距就可以小於尺寸下限或大於尺寸上限。
`element_size` 仍是候選間距的偏好。實際分段還要配合必要接點、共用接線的奇偶與 Quad4 配對。
若分區混有普通邊，整個分區都套用已指定的尺寸比例。

可重採樣的完整密集 LINE 會沿厚度對半分。兩條側邊各有三點，中點連成兩個
sublayer 共用的節點鏈。第一層用密集 LINE 的轉接規則將 source 邊合併到中線，第二層直接以
該中線為輸入，再轉接到外框，並保留外框的必要接點。`max_transition_count` 在每層各自生效。
例如在 `debug_imprint_line.py`，厚度 400 分成兩層各 200，上限為 10 時，37 條 source 邊
先轉為 5 條中線邊，再轉為 1 條外框邊。若某條中線無法接到外框，會再試下一個中線候選。
若側邊的必要 pattern 節點與幾何中點衝突，建網會回報 section 和 seam ID。

若相鄰普通 LINE 因共邊中點而無法滿足奇偶，會將該 LINE 最長的可分割外框段對半分。
例如內側是兩段 50、外側是一段 100 時，外側會補成兩段 50。新增節點會出現在
`result.outline` 和該 section 的 `parity_split_node`，其他間距仍依原規則檢查。
固定外框模式沿用單層接合。

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
- 相鄰兩段會先比較相向端點的候選位置，吸附既有節點也計入。若外框間距低於來源間距的
  `dense_min_gap_ratio` 倍，兩端都改走法線，背向端仍可展寬。剛好達到門檻時允許展寬。
  比例設為 `0` 可沿用原行為，設為 `1` 則保留完整間距。固定外框若缺少法向節點，會略過該端。
- 一般模式先確定各端點吸附的既有接點。其餘端點從法線投影位置沿外框向密集段外側移動
  `d × tan(angle)`，其中 `d` 是該端點到外框的實際法向距離。若展寬會越界、交叉或使相鄰分區退化，則保留合法的法線投影。角度為 0° 時採用法線投影。
- 有一端展寬時，中線和外框的自由轉接點保留來源點在區間內的相對位置，必要外框接點仍固定。
  例如密集段寬 20、層厚 10、角度 45°，兩端都需要新增時，外框寬 40。來源位置在 20% 的轉接點距新左端為 8。
- 固定模式只使用 `outline_o` 的既有 node。一般模式新產生的接點也不會成為其他端點的吸附候選。

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
`max_added_elements`、`max_transition_count`、`min_size_ratio`、`max_size_ratio`。
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
    # 如需密集段分界，可設定 small_element_size、small_element_group_count 與 dense_mapping_angle。
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
| `line_policy_active` | 本層是否允許 LINE 重採樣，固定外框為 `False` |
| `max_transition_count`、`min_size_ratio`、`max_size_ratio` | 傳入並驗證後的值 |
| `small_element_size`、`small_element_group_count`、`dense_mapping_angle` | 短邊長度、連續段數門檻與密集段映射角度 |
| `dense_min_gap_ratio` | 相鄰密集段的最小保留間距比例 |
| `dense_sections`、`dense_seams` | 成功形成的完整密集分區與新增分界；單端切分可能留下混合 LINE |
| `dense_mappings` | 每個額外端點的 `endpoint`、`status`、`reason`；採用時另含 target ID／座標、角度及 seam ID |
| `corner_mappings` | 固定模式自由轉角接點的來源／目標 ID、替代狀態與候選順位 |
| `added_nodes`、`added_elements` | 本次新增總量，包含 repair |
| `repair_added_elements`、`expand_added_elements` | repair 與後續轉接各自新增量 |
| `edge_expansion`、`snapped_patterns` | 實際厚度與自動擴展採用的 pattern；指定外框時後者皆為 `None` |
| `section_details[].max_alignment_angle` | LINE 內部接線相對外框法線的最大偏角，單位為度 |
| `section_details[].size_band_active` | 該 LINE 是否實際套用至少一側尺寸限制；兩個比例皆為 `None`、完整密集區或固定外框時為 `False` |
| `section_details[].midline_ids` | 完整密集區的兩個 sublayer 共用的中線節點 ID，依左至右排序 |
| `section_details[].sublayer_max_transition_counts` | 兩個 sublayer 各自每條輸出邊承接的最大輸入邊數，依內、外層排序 |
| `section_details[].max_actual_transition_count` | 每條最終外框邊承接的原始 source 邊數的最大值。密集區的跨層統計可大於單層上限 |
| `section_details[].parity_split_node` | 相鄰 LINE 為滿足奇偶額外插入的外框節點 ID，只在補點時出現 |
| `section_details[].min_outer_spacing`、`max_outer_spacing` | 該 LINE 最終外框的最小、最大間距，採用原模型長度單位 |
| `quality` | 新元素最差長寬比、最小 scaled Jacobian 等品質資訊 |

`dense_mappings.status` 為 `mapped` 或 `skipped`。
`gap_limited` 表示該端是否因相鄰密集段的間距而限定在法線方向。
`reason` 包含吸附既有點的 `existing_contact`、向外展寬的 `expanded_contact`、
沿法線新增的 `projected_contact`、完整建網重用接點的 `canonical_contact`，以及
`no_eligible_node`、`no_valid_projection`、`not_planned`。
預覽時需要新增的接點，其 `target_node` 為 `None`。

完整網格 API `generate_full_adaptive_mesh()` 也接受 `small_element_size`、`small_element_group_count`、`dense_mapping_angle` 與 `dense_min_gap_ratio`。
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
| `line_constraints_infeasible` | 檢查該 LINE 的必要接點、轉接數及奇偶限制。一般及混合 LINE 也須檢查尺寸帶 |
| `element_limit_exceeded` | 檢查 `limit`、`required_elements`，調整預算或上游尺寸設定 |
| `topology_change` | 外框穿過 source、接到已佔用區域，或形成懸掛節點／非流形接合 |
| `numeric_range_exceeded` | 檢查 pattern 範圍與座標尺度 |
| `numeric_resolution_exceeded` 且 `stage="pattern_normalization"` | 檢查 `pattern_index` 指出的 guide 與 `source_support`。當多條邊的間距落在容差臨界值，選用的共同座標系可能無法一致判定所有接觸 |

演算法不會透過縮小整層厚度、插入三角形或退化元素來掩蓋失敗。

## 12. 舊版參數遷移

`dense_region_tolerance` 已移除，不保留相容別名；繼續傳入會得到 `TypeError`。
請改用 `dense_mapping_angle=30.0`，單位為度，縮放模型時不需隨長度縮放。
舊版距離容差不能直接當作新角度值使用。

新規則還限制 begin／end 的偏移方向，因此原本距離很近但朝密集段內側的 node 不再被選中。
`small_element_size` 名稱保持不變。未指定 `fixed_outer_boundary` 時仍允許外框補點。

## 13. FEM 交付回歸比對

- `tests.unit.mesh2d.test_adapter_fem_regression` 以同一個非均勻矩形測試兩組尺寸比例，分別核對
  元素連接表與外框 ID，再以 2×2 Gauss 積分求解單位熱源、單位熱傳導係數、外框溫度為零的穩態熱傳。
  當候選接縫的評分只差浮點捨入誤差時，依固定順序選擇。這項修正使預設網格在原始節點 `(24, 20)`
  的參考溫度由 `130.50917411746101` 變為 `129.09546941727265`。顯式使用
  `min_size_ratio=0.5, max_size_ratio=1.25` 時為 `128.05851863492813`。
  測試以長度倍率 `1`、`0.01` 和 `25.4` 重建網格，檢查連接表、總熱源與線性溫度場的內點殘差，
  並在換回原單位後以絕對允差 `1e-8` 比對測點溫度。
- `tests.integration.mesh25d.test_case1_tolerance` 使用 Case1 原始輸入檢查八個介面、材料體積與 CDB 匯出。
  在中間層與最後一層都使用表中設定時，兩組皆保留八個共形介面，材料 `Si` 與 `si` 的體積分別為
  1,240,000,000,000 與 79,000,000,000，匯出座標逐值回讀一致。

| Case1 的 LINE 尺寸比例 | 修正後節點數 | 修正後元素數 | 修正後完整 CDB 的 SHA-256 |
| --- | ---: | ---: | --- |
| 預設 | 319,133 | 232,202 | `ce332b7f19a598117555082e5a5750deb3922ec60231d2c07752d1c7791ff825` |
| 下限 0.5、上限 1.25 | 317,910 | 231,195 | `e9339de1925f405e157149fa618fb2259ad1bc55d9c867a07d8e403a8181d38f` |

```sh
PYTHONPATH=src venv/bin/python -m unittest \
  tests.unit.mesh2d.test_adapter_fem_regression tests.integration.mesh25d.test_case1_tolerance
```

修正前的預設網格有 319,139 個節點、232,229 個元素，CDB 簽章為
`d839c6a3ae4dd320545c9bf3b9eee865a2d1c5f1093a73cf36d65d621c5cd157`。
舊尺寸比例在修正前的簽章為 `94020591c1f0a094ac4ac7634b756677c55063c18e6da27c87db89da3ddb989d`，
與 `3d5417c` 相同。當時該網格和提交 `8b534a8` 的元素連接表相同，
有 8,724 個節點移位，最大位移 75 µm。舊案的物理結果應以交付時封存的網格、
求解器設定及測點結果為比對基準。

密集轉接也需要檢查形狀品質。在專案的 dense benchmark 中，使用 `element_size=1000`、
`small_element_size=10` 與現行預設尺寸比例時，轉接上限 20 的新增 Quad4 最小
scaled Jacobian 約為 0.02893，最大邊長比約為 115.47。將 `max_transition_count`
提高到 1000 時，最小 scaled Jacobian 降至 0.00305，最大邊長比增至 316.42。
