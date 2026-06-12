# 近期质量提升与性能优化总结（2026-06-05 ~ 2026-06-12）

更新时间：2026-06-12
范围：最近 7 天（50 个 commit）实际被 main 分支采用的所有质量提升和性能优化策略。
不包含被回滚的 commit（见 §10）。
面向读者：负责 lip-sync 质量迭代、性能调优、运维回归测试的工程师或 AI coding agent。

本文是过程文档（"过去一周做了什么"），不是策略设计文档。策略语义请参考：
- 切段/合并：[heygen_like_lipsync_segmentation_td.md](./heygen_like_lipsync_segmentation_td.md)
- 调参方法论：[AGENTS.md §"Tuning Methodology & Lessons Learned"](../AGENTS.md)

---

## 0. 一句话总结

视频的"嘴部动、画面不跳"是这一周的主线。所有策略最后都收敛到 3 件事：
1. **跨帧时序约束**（output_temporal_blend + CodeFormer EMA）抹掉 per-frame 闪烁
2. **段一致性切分**（majority vote + time-window merge + hard cut + track-aware）让"送模/不送模"的决策在段级稳定
3. **嘴部融合保真**（lips_only mode + mouth_color_match + lips dilation）让合成嘴和原脸色调、边界都对齐

性能侧重点是减少重复计算（BiSeNet 缓存、PNG 压缩、numpy fast-path）。

---

## 1. 输出级时序平滑（output_temporal_blend）

| commit | 改动 |
|---|---|
| `e6f22c6` (06-09) | 引入 `output_temporal_blend` 旋钮（opt-in，默认 0.12） |
| `5374be1` (06-11) | **默认 0.12 → 0.25**；mouth_delta 门控从硬切 0.16 改成 smoothstep taper（0.10 满、0.30 关） |

**策略**：
- 在 `_write_result_frames` 的渲染循环里，对每帧的 `resized` 做 `cv2.addWeighted` 与上一帧的 `resized` 混合
- α=0.25（当前默认）= 75% 当前帧 + 25% 上一帧；时间常数 ≈ 4 帧 ≈ 0.13s @ 30fps
- **关键改进（5374be1）**：用 mouth-region delta 的 smoothstep 替代硬切。硬切 0.16 在 'P' → 'AH' 之类真实嘴部动作时会被触发，**正好在你最需要平滑的时候把平滑关了**。smoothstep 让"嘴部小动"持续享受平滑，只在头部大转/切镜头时跳过。

**和 CodeFormer EMA 的关系**：两者是**串联**的两层平滑（CodeFormer 1 阶 EMA α=0.8 + output 1 阶 blend α=0.25）。两层都开着会双重低通滤波。详细讨论见 §6 末尾的"组合效应"。

---

## 2. 段一致性切分（segment_consistency）

| commit | 改动 |
|---|---|
| `63012ec` (06-11) | 引入"per-segment all-or-nothing"：段内混合 passthrough/generated 时整段降级到 passthrough |
| `e396ebf` (06-11) | **all-or-nothing → majority vote**：passthrough > 50% 才降级；ties 保留 generated |
| `aa8a653` (06-11) | 引入"时间窗口合并"：相邻 valid run 间距 < `merge_window_seconds` (1.0s) 视为一段；吸收 detector 抖动 |
| `4b4987a` (06-12) | 引入 3 个新门：`hard_cut_enabled` / `track_aware` / `min_merged_seconds` (1.5s) |
| `ce7b684` (06-12) | 配套：CodeFormer EMA 在 track_id 切换时主动拒绝 mix（避免把旧身份平滑到新身份） |

**策略演化**：

```
63012ec: 段内 1 个 passthrough → 整段降级
              ↓ 太激进
e396ebf: 段内 passthrough > 50% 才降级，ties 保留 generated
              ↓ detector 抖动把短段切碎
aa8a653: 时间窗口合并 1.0s 内相邻 valid run
              ↓ 硬切/角色切换被错误合并
4b4987a: 3 个新门
  - hard_cut_enabled: gap 内人脸直方图距离 > 0.65 → 禁止合并
  - track_aware: face-crop 连续性 + 段级 track_id → 不同 track 禁止合并
  - min_merged_seconds=1.5: 合并后时长 < 1.5s → 整段 passthrough
              ↓ EMA 链在 track 切换处 pop
ce7b684: CodeFormer EMA 也读 track_id，track 切换时拒绝 mix
```

**doc 文档**：[heygen_like_lipsync_segmentation_td.md §5.1 §5.5 §5.7](./heygen_like_lipsync_segmentation_td.md)

---

## 3. 嘴部融合模式（lips blend modes）

| commit | 改动 |
|---|---|
| `214cffd` (06-10) | `get_image_prepare_material` 新增 `lips_only` 模式 |
| `7fd6373` (06-10) | `/api/lipsync` validator 接受 `parsing_mode='lips_only'` |
| `bff3cb8` (06-10) | 新增 `mouth_color_match_strength` Pydantic 旋钮（初值 0.45） |
| `c771614` (06-10) | 串到调用点 |
| `d68a302` (06-10) | 更紧的 lips blend + `lips_outer_only` 模式 |
| `87f9884` (06-10) | **`lips_blend_dilation` 默认 0 → 2**（防止嘴部边界剪切） |
| `6b0e700` (06-10) | paste 时用 expanded-crop lips mask，避免圆形 upscale 边界 |
| `62ceb85` (06-10) | lips_mask 在 paste 前 resize 到 crop_box 尺寸 |
| `3c4ca83` (06-10) | lips-only mask 用于 blend / color match / CLAHE |
| `d3fdba6` (06-10) | **`mouth_color_match_strength` 默认 0.45 → 0.30**（保留唇部饱和度） |
| `a72390e` (06-10) | 去掉 mouth CLAHE，加强 mouth color match |

**当前默认组合**：
- `parsing_mode='jaw'`（用户可改 `'lips_only'` / `'lips_outer_only'` / `'raw'`）
- `lips_blend_dilation=2`
- `mouth_color_match_strength=0.30`
- `mouth_sharpen_strength=0.50`
- `mouth_detail_strength=0.90`

**与 CodeFormer 的耦合**：嘴唇饱和度、嘴部细节、边界锐利度这三件事现在交给融合阶段处理（mouth_color_match + sharpen + detail_restore），不再依赖 CodeFormer 模型去做细节恢复——所以 CodeFormer 的 fidelity 可以拉到 0.85 而不丢细节。

---

## 4. 渲染质量门控（quality gates）

| commit | 改动 |
|---|---|
| `1aa48fb` (06-10) | 4 个 badcase gate 一次性引入：laplacian / sharpness ratio / face-tile MSE / upper-face color histogram |
| `1a6d93c` (06-09) | mouth-region laplacian gate（默认开） |
| `b022880` (06-09) | 轻度 color histogram check（默认开） |
| `1abcab1` (06-09) | drift 阈值下调 + per-tile max-MSE gate |
| `9a82feb` (06-09) | detail restore 限制到 upper face + drift fallback |
| `9177288` (06-09) | color match 只作用 upper face，不污染嘴部 |
| `c284ca2` (06-09) | upper-face 后处理用 soft transition（线性 ramp） |
| `c2d2429` (06-10) | reference detail restore 限制到 skin |
| `58eab08` (06-10) | CLAHE 嘴部色块修复 |
| `e03882c` (06-10) | generated mouth 颜色向 reference mouth 对齐 |
| `ba37373` (06-10) | 恢复 hard-cut color match + 整脸 MSE |
| `68b7fb1` (06-09) | 放宽 drift + tile MSE 阈值（300→700, 5→X） |
| `b822dab` (06-10) | per-tile MSE fallback 默认开 |
| `cc51e3c` (06-10) | 回滚过于严格的质量门控，重新平衡 |
| `72cd85c` (06-10) | **质量门控默认全关，作为 opt-in 调试旋钮** |

**关键设计原则**（来自 [AGENTS.md](../AGENTS.md) "Filter tuning"）：
- **改变方法，不只改变阈值**：上调阈值是错的，要先看 per-gate 计数器是哪个在跳
- **门控的语义要对**：identity matching 答"改谁"、speech activity 答"改不改"，**单门不混答**
- **CHISQR on upper-face + Laplacian on mouth** 比 whole-face MSE 更能抓到局部色块

**当前默认**：`quality_gate_enabled=False`（opt-in）。生产默认开反而容易误判。

---

## 5. 人脸跟踪（face lock）

| commit | 改动 |
|---|---|
| `604b11f` (06-09) | 引入基于 IoU 的 face lock（防止切到错误的相邻人脸） |
| `2bcda08` (06-09) | **IoU → center-distance 比率**（默认 2.0 face-widths），更鲁棒 |
| `0e89728` (06-09) | "soft" lock：lock 失败时 fallback 到 most-open-mouth 候选 |
| `62a3923` (06-09) | bbox 平滑 + 跳过侧脸 + 加宽 margin |

**与 HiRes 输出的关系**：bbox IoU 在小角度旋转/缩放变化时容易碎。center-distance 相对 face-width 归一化，对 in-plane 运动更鲁棒。

**关键设计**（[AGENTS.md](../AGENTS.md) "Face lock design"）：
- "soft" lock 必须**真实现** fallback：同时记录 best_locked 和 best_unlocked，prefer locked，fall back unlocked
- Lock 旋钮默认 0（opt-in）；默认行为是"宽松"
- "speaking moves, silence stays still" 在 lock 内依然适用：locked 目标可见但音频静默 → passthrough

---

## 6. CodeFormer 集成与稳定化

| commit | 改动 |
|---|---|
| `501737f` (06-05) | 把 CodeFormer 接入 MuseTalk pipeline |
| `44a9575` (06-05) | 人脸 resize 到 512x512 喂模型 |
| `22a1865` (06-05) | fidelity 默认 0.5 + mouth-region pixel diff fallback |
| `3e92ae1` (06-10) | 引入 1 阶 EMA（α 默认 0.8）+ fidelity 0.5 → 0.7 |
| `90db126` (06-10) | **EMA 只对相邻 generated frame 混合** + fidelity 0.7 → **0.85** |
| `ce7b684` (06-12) | EMA 在 track_id 切换时拒绝 mix + 新增 `ema_chain_breaks` / `ema_resets_on_track_switch` 计数器 |

**当前默认**：
- `codeformer_fidelity_weight=0.85`（比 README 的 0.5 保守）
- `codeformer_temporal_alpha=0.8`（EMA 80% 当前 + 20% 上一帧）
- `codeformer_adain=True`
- `codeformer_enabled=False`（用户显式开启）

**wrapper 内部 fallback**（`codeformer_restorer.py` 硬编码）：
- `fallback_sharpness_low=0.5` / `fallback_sharpness_high=2.0` / `fallback_pixel_diff=0.20` / `fallback_mouth_diff=0.15`
- 任一指标超阈 → 把单帧 restored crop 替换为原 inpainted crop
- 触发后该帧进 EMA 链 → 可能产生 pop

**与 §1 output_temporal_blend 的组合效应**：
- 单层（CodeFormer α=0.8）= 截止频率 0.16 cycles/frame
- 串联后 = 两层低通叠加，截止频率更低
- 副作用：快速嘴部动作（'P' → 'AH' 之类）有 1-2 帧延迟，**视觉上像"跳"**
- 缓解办法（已 commit 的）：两边 EMA 都尊重 `idx - prev == 1` 规则 + 双方都读 `track_id` 在切人时拒绝 mix
- 进一步缓解（待做）：如果两层同时开，可自动把 `output_temporal_blend` 上限收紧到 0.1（**未实现**）

**未实现的下一步**：
- B：暴露 wrapper fallback 4 个阈值为 `LipSyncRequest` 字段（默认放宽：pixel_diff=0.30 / mouth_diff=0.25）
- E：如果两层同时开，自动收紧 `output_temporal_blend` upper bound

---

## 7. /api/faces 静音脸过滤

| commit | 改动 |
|---|---|
| `8a382f0` (06-11) | `/api/faces` 新增 `min_mouth_openness`（默认 0.10） |

**策略**：
- 用 HSV 嘴部区域的 V < 80 像素比例作为"嘴张开"指标
- 0.0=闭嘴, 0.17+=明显张开
- per-cluster 取"采样帧中最大开口度"，只要一帧嘴张开过就保留
- 避免 listener / side-glance / 背景人物被误识别

**向后兼容**：旧客户端不传字段也能用，但默认 0.10 会开始过滤静音脸——如果有依赖"返回所有人脸"的工作流要显式传 `min_mouth_openness=0.0`。

---

## 8. 性能优化

| commit | 改动 | 量级 |
|---|---|---|
| `62e225a` (06-10) | BiSeNet parsing per-frame 缓存 | +368 / -222（最大单 commit），重复 source frame 不再跑两次网络 |
| `7b8c49d` (06-10) | PNG writes `IMWRITE_PNG_COMPRESSION=1` + `get_image_blending` numpy fast-path | +56 / -15，写盘耗时减半 |
| `67c2f7c` (06-10) | 用 skin mask 做 color match + soft mask blend | +48 / -26，色彩匹配只在 skin 区域跑 |

**当前 25fps / 5s 视频（~125 frames）**：
- 之前：~4-5s 总耗时（3-4s 在 BiSeNet 重复 forward + 0.5s 在 PNG 压缩）
- 之后：~2-3s 总耗时（BiSeNet 减半 + PNG 减半 + 色彩匹配范围缩小）

具体数字需要在服务器 GPU 上 profile，本地估算见 [AGENTS.md "Repository Runtime Notes"](../AGENTS.md)。

---

## 9. 调参方法论（AGENTS.md 新章节）

| commit | 改动 |
|---|---|
| `8e6a3db` (06-10) | AGENTS.md 新增"Tuning Methodology & Lessons Learned"章节 |

**核心要点**（摘要，完整版见 [AGENTS.md](../AGENTS.md)）：
1. **Filter tuning**: 改方法 > 改阈值；先看 per-gate 计数器是哪个在跳
2. **避免色块/边界伪影**: 硬切 mask → 软切；color match + detail restore 混合而非切换
3. **Face lock 设计**: soft lock 必有 fallback；lock 旋钮默认 0；"speaking moves, silence stays still" 在 lock 内仍适用
4. **帧稳定性检查**: bbox 平滑只锁位置不锁内容；要 output-level temporal blend
5. **CodeFormer 是可选**: loader 失败 → degrade gracefully；不强制
6. **MuseTalk 是 encoder-decoder, 不是扩散**: 不要再加 diffusion-only 字段

---

## 10. 没被采用的（rollback + cleanup）

| commit | 内容 | 状态 |
|---|---|---|
| `872118f` (06-10) | 放宽 CodeFormer internal fallback + 加强后处理 | **被 `16a44f6` 回滚**——发现"放宽"导致 flicker 反而严重 |
| `16a44f6` (06-10) | 撤销 `872118f` | ✅ 保留 |
| `2851bde` (06-10) | 清理"近期质量迭代"中产生的死代码 | ✅ 保留 |

**学习点**：CodeFormer 周边修改迭代很快（一天 5-6 个 commit），放宽/收紧经常要回滚；不要假定"前一个 commit 是对的"。

---

## 11. 关键数值变更（commit 前后对照）

| 旋钮 | 一周前 | 现在 | commit |
|---|---|---|---|
| `output_temporal_blend` | 0.12 | **0.25** | 5374be1 |
| `output_temporal_blend` mouth-delta 门 | 硬切 0.16 | **smoothstep 0.10→0.30** | 5374be1 |
| `codeformer_fidelity_weight` | 0.5 | **0.85** | 3e92ae1 → 90db126 |
| `codeformer_temporal_alpha` | (无) | **0.8** | 3e92ae1 |
| `lips_blend_dilation` | 0 | **2** | 87f9884 |
| `mouth_color_match_strength` | 0.45 | **0.30** | d3fdba6 |
| `quality_gate_enabled` | True | **False** | 72cd85c |
| `segment_consistency_merge_window_seconds` | (无) | **1.0s** | aa8a653 |
| `segment_consistency_hard_cut_enabled` | (无) | **True** | 4b4987a |
| `segment_consistency_hard_cut_distance_threshold` | (无) | **0.65** | 4b4987a |
| `segment_consistency_track_aware` | (无) | **True** | 4b4987a |
| `min_merged_lipsync_seconds` | (无) | **1.5s** | 4b4987a |
| `FaceListRequest.min_mouth_openness` | (无) | **0.10** | 8a382f0 |

**总开关**：`codeformer_enabled` 默认 `False`，`quality_gate_enabled` 默认 `False`。其余**默认即"开"**。

---

## 12. 已知未解决问题（carryover）

| 问题 | 状态 | 建议下一步 |
|---|---|---|
| CodeFormer + output_temporal_blend 双层叠加导致快速嘴部动作 ghosting | 双方 EMA 都读 track_id 缓解了一部分 | 下一轮做 E（自动收紧 `output_temporal_blend` upper bound） |
| CodeFormer wrapper 的 fallback 阈值硬编码，触发时单帧 pop 进 EMA 链 | 不可控 | 下一轮做 B（暴露 4 个阈值为 `LipSyncRequest` 字段） |
| `target_fill_*` 一族旋钮在段合并（aa8a653 + 4b4987a）后的相互作用未充分验证 | 暂无 | 服务器端用含多人/硬切的视频回归 |
| `0e89728` 的"soft face lock fallback" 失败时偶尔仍然跳过整个说话段 | 暂无 | 下一轮在 `_filter_motion_targets` 加 lock 状态 log |

---

## 附录 A：相关文档链接

- 切段/合并策略设计：[heygen_like_lipsync_segmentation_td.md](./heygen_like_lipsync_segmentation_td.md)
- 调参方法论：[AGENTS.md §"Tuning Methodology & Lessons Learned"](../AGENTS.md)
- 项目结构总览：[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)
- API 行为约束：[AGENTS.md §"API Surface"](../AGENTS.md)
