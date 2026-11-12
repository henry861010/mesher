# Process-flow UF 共點與拓樸保護

`PolygonGeometry` 的不同 loops 可以共用原本就存在、XY 座標完全相同的頂點。
此規則適用所有 process-flow polygon，不依賴 `UF` 的 material 或 key。
檢查位於 export 的 `geometry_normalization`；kernel UF outline、body 數量、
schematic preview、schema 與預設 `geometryTolerance=0.001` 都不變。
Mesh sizing region 也使用同一個拓樸 helper，但維持原有 loops 必須分離的契約；
本次沒有放寬 sizing region 的可接受形狀。

## 接受條件與保護

- 每個 loop 自身有效；兩個邊界的完整交集只能是 Point 或 MultiPoint，
  且所有交點都必須是雙方既有頂點。不能以 tolerance 將近點視為共點。
- 共邊、交叉、部分面積重疊、單一 loop 自我接觸維持拒絕。
- 依包含關係與深度奇偶組成 shell／holes。對有接觸的區域驗證完整組合，
  包含所有 sibling holes，防止多個點接觸共同切斷區域內部。
- normalization 前後同時比較包含關係與 `(loop, loop, vertex, vertex)` 接觸身分。
  接觸座標可以隨既有 bounded normalization 移動，但接觸不能新增、消失或換頂點。
  可保持有效卻改變接觸身分時回報 `feature_below_resolution` /
  `polygon_contact_change`；共邊等無效結果仍回報 `polygon_loop_contact`。
- 沒有接觸、但兩個 loop 的正距離不大於 T 時，仍回報 `polygon_loop_gap`。
  本次未新增局部 tiny gap／short feature 修復，既有 edge-collapse 保護保留。

使用既有 STRtree 候選篩選；只有零距離候選才執行求交、建立可重用的頂點索引，
以及檢查受影響 shell 的 holes。沒有 polygon 切分、全頂點兩兩搜尋或新的 preview 工作。

## 回歸與重現

新增測試使用合成六顆 die，直接執行 kernel `apply_under_fill` 產生共點 `(10, 16)`。
保留一個 UF body、兩個 loops；完整 UF 面積 360 µm²、厚度 6 µm、體積 2160 µm³。
測試核對每種材料的體積、XY 覆蓋、UF hole、不變的輸入資料，以及 worker CDB 讀回結果。
Adapter 案例使用涵蓋共點的實際 refinement region，並檢查 `adapterRegions`，避免
僅指定 adapter 但實際退回 grid。四種 symmetry 模式另以 grid 驗證裁切後體積與覆蓋。

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/mesher-uf-mpl python -m unittest \
  tests.unit.mesh25d.test_polygon_contacts \
  tests.integration.mesh25d.test_tolerance_policy \
  tests.integration.mesh25d.test_uf_shared_vertices -q
```

## 效能重現

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/mesher-uf-mpl python \
  benchmarks/mesh25d/benchmark_polygon_contacts.py \
  --baseline-ref 79e933a --repeat 10 --output /tmp/uf-polygon-contacts.json
```

每組先暖機 2 次，再交替執行 baseline／candidate 各 10 次，記錄 median、p95 與原始樣本。
分別量測完整 normalization 與 `build_mesh → CDB write`；後者包含 normalization，
不含 worker process 啟動。fixture 產生、獨立的輸入複製和正確性比較不計入時間。
所有規模都量測 normalization；完整 export 預設只量測不超過 600 個頂點的代表案例，
避免既有 3D 材料搜尋成本主導大型測試。可用 `--max-export-vertices` 提高此上限。
模型是合成 UF polygons 加基板，以隔離頂點／holes／共點數的影響；未載入 saved flow
或資料庫，不代表公司模型的完整執行時間。

一般無共點案例的 normalization 增量門檻為 `max(基準中位數 × 5%, 2 ms)`。
共點案例的 baseline 本來就會失敗，只記錄新版成功路徑的成本，不拿失敗時間作比較。

## 驗收紀錄（2026-11-12）

基準為 mesher `79e933a8dce881f0df753910f60f59952685620d`；使用 process-flow 開發環境
Python 3.13、Shapely 2.1.2。完整 discovery 共 **1,217 項測試通過**，耗時 158.665 秒。
其中新增 12 項拓樸／sizing region 測試與 2 項 UF／worker 整合測試；單項測試中的參數案例另涵蓋
loop 順序、方向、起點、閉合表示、共點群、四種 symmetry 與實際 adapter refinement。

主專案的 CDB worker bridge 另有 2 項測試通過，API 初始化使用記憶體資料庫。
原始效能樣本、程式 SHA-256 與環境資訊保留在 [benchmark JSON](uf-shared-vertices-benchmark.json)。

以下為同機中位數，單位 ms；`—` 表示未量測。四組無共點案例均通過 normalization 增量門檻。

| 案例 | 頂點 | 共點 | 舊 normalization | 新 normalization | 舊 build→CDB | 新 build→CDB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 一般 | 140 | 0 | 4.337 | 4.321 | 51.400 | 51.539 |
| 一般 | 572 | 0 | 19.378 | 19.387 | 578.864 | 579.840 |
| 一般 | 2300 | 0 | 85.767 | 86.007 | — | — |
| 一般 | 9212 | 0 | 398.866 | 398.702 | — | — |
| 共點 | 20 | 1 | — | 0.933 | — | 4.584 |
| 共點 | 500 | 25 | — | 15.419 | — | 847.417 |
| 共點 | 2000 | 100 | — | 59.064 | — | — |
