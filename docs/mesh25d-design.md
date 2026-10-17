# Mesh25D 架構與資料流

`mesher.mesh25d` 直接接收 Standard V1 geometry dict 與 mesh control dict，整合
2D mesh 與 Dragger。公開入口為 `build_mesh(geometry_structure, mesh_control,
progress=...)`，回傳 `Mesh3D`；各階段的 plan 與 builder 為內部介面。

## 模組責任與依賴

實線表示呼叫或依賴，虛線表示只引用底層的控制資料型別。

```mermaid
flowchart TB
    USER["Python 呼叫端"]
    PF["mesher.process_flow<br/>worker CLI・progress 傳輸・CDB 輸出"]
    M25["mesher.mesh25d<br/>build_mesh(geometry_dict, mesh_control_dict)<br/>整合 geometry、2D 與 3D 階段"]
    MC["mesher.meshcontrol<br/>reference resolution<br/>各 control method 的控制語意"]
    M2["mesher.mesh2d<br/>XY control 資料介面<br/>grid・circular 演算法"]
    M3["mesher.mesh3d<br/>Z／layer 資料介面<br/>Dragger"]
    CONTRACT["mesher.contracts<br/>JSON 輸入格式驗證"]
    USER --> M25
    PF --> M25
    M25 --> MC
    M25 --> M2
    M25 --> M3
    M25 --> CONTRACT
    MC -. "只引用控制資料型別" .-> M2
    MC -. "只引用控制資料型別" .-> M3
```

`mesh25d` 的內部責任分為 geometry、plan2d、plan3d 與 pipeline：

- `plan2d.py`：規劃格線與圓形操作，產生 `Mesh2D`。
- `plan3d.py`：規劃 Z layers，透過 Dragger 產生 `Mesh3D`。
- `pipeline.py`：串接 geometry、controls、2D 與 3D 階段。

底層 `mesh2d`、`mesh3d` 不反向依賴 `mesh25d`、`meshcontrol` 或 `process_flow`。
共用 control 與資料介面不載入 Dragger 或 optional meshing runtime。

## 建網格資料流

```mermaid
flowchart TB
    G["geometry dict<br/>Standard V1"]
    C["mesh control dict"]
    PREP["驗證輸入・深拷貝<br/>依 symmetry 保留幾何"]
    G --> PREP
    C --> PREP
    OBJ["meshcontrol<br/>解析 reference → 建立 control 物件"]
    PREP --> OBJ
    XY["各 method：get_25D_xy_controls()<br/>回傳區域＋目標尺寸"]
    Z["各 method：get_25D_z_controls()<br/>回傳已決定分段的 Z 區間／點"]
    OBJ --> XY
    OBJ --> Z
    subgraph PLANAR["mesh25d.plan2d：2D 階段"]
        PATTERN["Geometry → Pattern2D<br/>footprints・KOZ・來源資訊"]
        CHECK{"XY controls 是否為空？"}
        P2["2D planner → Plan2D<br/>格線・圓形支援線・操作順序"]
        A2["呼叫 mesh2d 演算法<br/>grid → circular imprint／extend"]
        MESH2["Mesh2D"]
        PATTERN --> P2
        CHECK -- "是：目前 Z methods" --> P2
        P2 --> A2 --> MESH2
    end
    subgraph SOLID["mesh25d.plan3d：3D 階段"]
        ASSIGN["Geometry → assignments<br/>_get_assignments 內容完全保留"]
        LAYER["Z layer planner → LayerPlan<br/>合併邊界・控制區間・assignments"]
        ADAPTER["3D adapter<br/>2D winding 轉換・傳入 layer_infos"]
        DRAG["呼叫 mesh3d.Dragger"]
        MESH3["Mesh3D"]
        ASSIGN --> LAYER
        LAYER --> ADAPTER
        ADAPTER --> DRAG --> MESH3
    end
    PREP --> PATTERN
    PREP --> ASSIGN
    XY --> CHECK
    Z --> LAYER
    MESH2 --> ADAPTER
    CHECK -. "否：3D_REGION" .-> RESERVED["本次明確報錯<br/>XY 投影資料已定義<br/>區域加密演算法待後續實作"]
    MESH3 --> OUTPUT["process_flow<br/>CDB 輸出"]
```

## 控制與規劃的分工

- **Control method 決定控制語意。** TOP／BOT／CENTER 的餘段位置由各 class
  決定；planner 只使用回傳的 constraints，不依 method 名稱重新分段。
- **Planner 整合資料。** `Pattern2D` 保留 `BOX／CIRCLE／POLYGON` 與 `dim`；
  `Plan2D` 決定格線及 circular 操作；`LayerPlan` 合併 assignments 與 Z controls。
- **演算法執行規劃。** Grid、circular 與 Dragger 不解讀原始 mesh control JSON。

XY 使用端定義 `XYRegionControl`，包含 `region`（既有 2D face 格式）與
`element_size`。Z methods 回傳空的 XY controls。`3D_REGION` 只回傳絕對座標
XY 投影，不保留 Z 範圍、不增加 Z constraints；區域加密尚未實作，因此實際
建網格遇到非空 XY controls 時，須在 meshing 前回報 control index／label。

Z constraints 保留 `start`、`end`、`element_size`、`method`，其中 `method`
只作診斷資訊。`Z_POINT` 只增加邊界；重疊區間使用 global size 與有效 controls
的最小尺寸。控制範圍與模型 Z 範圍取交集，不延伸模型。

## Geometry、symmetry 與保護範圍

- 輸入先深拷貝；geometry extraction 不讀取 mesh control。
- Reference 必須在原始模型存在；symmetry 排除的目標，其 control 不生效。
  保留 container 的相對座標依保留子樹的 bounds 計算；多個符合目標仍各自展開。
- **`_get_assignments` 的名稱、參數與完整內容保持原樣，只搬移位置。**
  原有 priority、材料配置、KOZ、bump stage 與 assignment 行為須保留。
- Dragger 的分層、節點共用及 connectivity 行為保留；winding 轉換在 3D adapter。

## 遷移與驗證

Python 呼叫端改用 `mesher.mesh25d.build_mesh`；舊 builder／translator 與 control
轉換路徑移除。Worker CLI、CDB、metadata、progress 事件及目前 JSON contract
保持相容；不支援舊版座標格式。

驗證包含原始碼比對、無 controls 的 mesh／CDB 回歸、Z 分段與重疊、reference
及 symmetry、XY 投影與明確拒絕、輸入不變、import boundaries、完整測試及 build。
