# Case1 極短界面邊診斷

這份資料重現原始 2D adaptive mesh 的失敗，沒有套用修正。輸入快照的 `unitSystem` 是 `um`，所需絕對距離容差是 `0.001 μm`。

原始完整 case 的 plan 有 8 個 adapter regions，global grid 產生 2,064 nodes／1,904 elements。最後 fixed layer 在 `controls[7] ('Die top corners merged 3')`、`boundary_index=2` 的 `outline_o edge_index=9` 拒絕極短邊，與使用者 export log 相同。

| 項目 | 實測值 |
|---|---|
| Plan coordinate tolerance | `0.001` |
| 該 source 的 origin | `(-175, -10800)` |
| 該 source 的 scale | `2600` |
| Layer normalized tolerance | `2.842170943040401e-14` |
| Layer tolerance 換回世界座標 | `7.389644451905042e-11 μm` |
| Edge 9 世界座標 | `(2800.0000000000005,-8200)` → `(2800.0,-8200)` |
| Edge 9 原始世界座標距離 | `4.547473508864641e-13 μm` |
| Edge 9 normalized delta | `(-2.220446049250313e-16,0)` |
| Edge 11 世界座標距離 | `2.2737367544323206e-13 μm` |

Normalized delta 再乘回 scale 與原始世界座標直接相減有微小差異，因正規化也有浮點運算。本報告的「世界座標距離」均由原始世界座標計算；診斷另列 `normalized_delta_world_estimate`，避免混淆。

獨立 grid replay 觀察到：hole 2 的 support 是 `2800.0000000000005`，hole 5 是 `2800.0`。Builder 對 hole 5 的錨點延伸一條 `kind="geometry"`、axis 1、X=`2800.0`、Y 區間 `[-8200,7700]` 的 cut，於 hole 2 新增 `(2800.0,-8200)`，而該 hole 已有 `(2800.0000000000005,-8200)`。精確 set／cut key 不會把它們合併。

這個來源證據保存在 [grid_provenance.json](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/grid_provenance.json)；global generator 的原始輸入保存為 [grid_input.npz](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/grid_input.npz)。另外 hole 3／5 也有同類短邊，因此不能只跳過第一個報錯節點。

可用兩個完全軸向、彼此分離的矩形洞重現，不需要原案的複雜 controls 或 patterns：outer=`[0,5]×[0,5]`，hole A=`[1,nextafter(2,+∞)]×[1,2]`，hole B=`[1,2]×[3,4]`。Target size=1、coordinate tolerance=.001，仍輸出 `4.440892098500626e-16` 的 hole A 邊，並被 layer 的方向驗證拒絕。

從 repo root 執行：

```bash
PYTHONPATH=src venv/bin/python docs/tolerance-audit/case1/minimal_probe.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/mesher-tolerance-audit-mpl venv/bin/python docs/tolerance-audit/case1/diagnose.py
PYTHONPATH=src venv/bin/python docs/tolerance-audit/case1/grid_replay.py
```

[diagnose.py](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/diagnose.py) 預設使用同目錄輸入快照，也支援 `--case-dir`。它只在程序內包裝 `_prepare`、global generator 與 target normalizer，列印量化診斷；NPZ 快照寫到新建暫存目錄，或指定的 `--output-dir`。它捕捉預期錯誤後列印 traceback，程序 exit code 不能當作 case 成功判斷，應查看最後 `event=failure/success`。

實測結果：[diagnosis.jsonl](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/diagnosis.jsonl)、[traceback.log](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/traceback.log)、[minimal output](/Users/henry/Desktop/code/mesher/docs/tolerance-audit/case1/minimal_probe.output.txt)。

修正的驗收必須包含所有 region interfaces、正確的共享 ID／parity、完整覆蓋、沒有零邊或浮點殘差造成的 sliver，以及完成 3D extrusion 和 export。此處只重現目前在 2D 就中止的失敗，尚未宣稱 case 已修復。
