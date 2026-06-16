# Cross-project lessons: LatentSync → MuseTalk (2026-05-29 ~ 2026-06-12)

更新时间：2026-06-12
范围：LatentSync 近两周被 main 分支最终采用的所有改进策略，与 MuseTalk 当前实现做交叉对照。
目的：找出 MuseTalk 应该借鉴、已经做到、或者差异点不明显的策略，避免重复造轮子或重复踩坑。

相关文档：
- MuseTalk 近一周汇总：[quality_optimizations_2026_06.md](./quality_optimizations_2026_06.md)
- MuseTalk 调参方法论：[AGENTS.md §"Tuning Methodology & Lessons Learned"](../AGENTS.md)
- 两个项目的 API 字段名有意保持一致（"Field order is kept identical to LatentSync api.py:94-285 for cross-repository client compatibility"），所以下面很多旋钮名字直接对照。

---

## 0. 概览

| 类别 | LatentSync 已做 | MuseTalk 已做 | MuseTalk 待做 |
|---|---|---|---|
| 时序平滑 | ✓ EMA + 限域 + 切脸断 | ✓ output_temporal_blend + CodeFormer EMA | §A landmark EMA |
| 切脸断链 | ✓ 多处 | ✓ CodeFormer EMA (ce7b684) | §A output_temporal_blend 也断 |
| 侧脸 / yaw 过滤 | ✓ 完整实现（30° 阈值 + 段落 pad + 渐变区） | ✗ 接受参数不实现（兼容占位） | §B 实现 yaw 跳过 |
| CodeFormer 集成 | ✓ 3 tier 短剧配置 + 自适应 w + retry | ✓ 单 fidelity 旋钮 | §C 评估是否需要 tier |
| 嘴部 mask 来源 | ✓ landmark 动态 mask | ✓ BiSeNet parser 静态 mask | §D landmark 混合策略 |
| 测试 | ✓ 4 个测试文件 | ✗ 只有一个 ffmpeg smoke test | §E 补 unit tests |
| 性能 | ✓ DeepCache 2x speedup | ✓ BiSeNet cache + PNG fast-path | §F DeepCache 是否适用 |

下面按 §A ~ §F 逐个展开。

---

## §A. Landmark 派生量的时序平滑（MuseTalk 缺）

**LatentSync 做法**（`82e07f6` "Add EMA temporal smoothing to landmark-derived mouth_info"）：
- 在 AlignRestore 输出后，对 `center_x, center_y, half_width, half_height` 做 EMA（α=0.7）
- 目的：landmark 抖动会让 mask 边界也跟着抖；EMA 平滑掉 landmark 噪声后，mask 边界稳定
- 配套：`e56727d` 在人脸切换处断开 smoothing

**MuseTalk 现状**：
- DWPose landmark 直接喂给 `_landmark_bbox_for_face`，没做 EMA
- bbox 的稳定性靠后续 `_smooth_target_bboxes`（几何平滑）兜底
- landmark 自身的抖动会进入 bbox 计算，再被 bbox smoothing 抹平——**双层冗余但都不是从源头治**

**借鉴优先级**：🟡 中

**建议实现**：
- 在 `musetalk/utils/preprocessing.py` 的 `_pose_face_landmarks` 输出后加一个 `LandmarkEMA` 状态
- 对 `mouth_corners` (left/right) 和 `lip_top/bottom` 4 个点做 EMA（α≈0.7，对齐 LatentSync）
- **scope**：只对参与 bbox 计算的 4-6 个 landmark 做，不对全部 68 点
- **风险**：landmark EMA 在嘴部快速动作时会拖影；用 0.7 而不是 0.8 缓解

**代码改动预估**：~30 行，1 个 helper + 1 个 EMA state dict
**测试**：参考 LatentSync `test_temporal_continuity.py`

---

## §B. Yaw / 侧脸过滤（MuseTalk 占位未实现）

**LatentSync 做法**（多个 commit 形成完整方案）：
- `b28a8ae`: yaw_skip_threshold 45°→30°（更激进）
- `72b9c6b` / `0cde0a5`: 多信号 yaw 估计（nose + eye-width + mouth-corner），各自有 noise floor
- `f6b3652`: 段落 pad（episode pre/post_pad = 3 帧）
- `8b5fdbd`: 边界渐变区（cross-fade inpaint↔source，blend_fade_frames=3）
- `169fb86`: prev_yaw 不在 yaw-skip 段落间泄漏
- `b28a8ae`: 清理死代码

**MuseTalk 现状**（[api.py:595-608](../api.py)）：
```python
# Side-face / fast-turn prefilters (diffusion-only). MuseTalk does
# not currently implement yaw-based skipping; values are accepted
# for API compatibility and logged when non-default.
yaw_skip_threshold: float = Field(45.0, ge=0.0, le=90.0)
yaw_rate_skip_threshold: float = Field(28.0, ge=0.0, le=45.0)
side_face_episode_pre_pad: int = Field(0, ge=0, le=30)
side_face_episode_post_pad: int = Field(0, ge=0, le=30)
yaw_warn_threshold_ratio: float = Field(0.75, ge=0.0, le=1.0)
```

→ **5 个 yaw 相关旋钮全部占位**，实现见 [api.py:4491-4499](../api.py)（"yaw-based side-face filters ... MuseTalk accepts them but does not skip frames"）

**借鉴优先级**：🟢 高（如果客户视频含侧脸多，必做）

**建议实现**：
- 在 `_select_most_open_mouth_bbox` 输出前加一个 `_filter_by_yaw` 步骤
- yaw 估计复用现成 DWPose landmark：左/右眼角 + 鼻尖 → 三角法估算 yaw 角
- 跳过：yaw > `yaw_skip_threshold` → 该帧 bbox 置 None
- 段落 pad：连续 N 帧跳过 → 前后扩展 `pre_pad`/`post_pad` 帧也跳过
- 渐变区：边界前/后 `blend_fade_frames` 帧按 ramp 权重混合 inpaint + source
- **不需要新模型**——DWPose landmark 已经在跑，免费

**代码改动预估**：~150 行
**测试**：参考 LatentSync `test_yaw_filter.py` (25K)——这是 MuseTalk 缺的
**风险**：yaw 估计算法要校准；建议先抄 LatentSync 的实现再调整

---

## §C. CodeFormer tier 配置（MuseTalk 单旋钮，评估是否需要）

**LatentSync 做法**（`d3a4625` "CodeFormer short-drama tier-1/2/3"）：
- 3 档配置：tier-1 (轻度，w=0.9) / tier-2 (中度，w=0.7) / tier-3 (重度，w=0.5)
- 配套：自适应 w（按 frame quality 选档）+ retry（首帧失败时降档重试）+ mouth-only paste
- `3588f22` Color-match CodeFormer mouth ROI to inpainter（修复色块）

**MuseTalk 现状**（[quality_optimizations_2026_06.md §6](./quality_optimizations_2026_06.md)）：
- 单 `codeformer_fidelity_weight` 旋钮（默认 0.85）
- 单 `codeformer_temporal_alpha` 旋钮（默认 0.8）
- 没有 tier 概念

**借鉴优先级**：🟡 中

**建议**：
- 先观察：MuseTalk 当前默认 0.85 已经偏保守；如果大部分客户视频能直接跑通，**单旋钮够用**
- 如果发现某些长剧/短视频含明显退化，**再加 tier**：在 Pydantic 加一个 `codeformer_tier: Literal["off", "light", "medium", "heavy"]`，根据 tier 选 w 值
- **现阶段不做**，等 §B（yaw）和 §A（landmark EMA）落地后再评估

---

## §D. Dynamic per-frame inpaint mask from landmarks（架构差异，借鉴度低）

**LatentSync 做法**（`3774adf` / `dfdb390` / `451f68c` / `9cb71d4`）：
- 用 landmark 动态算出 inpaint region（U 形嘴部 mask）
- 给 UNet 用固定 U 形；给 restore 用 dynamic landmark 形
- 配套 kornia gaussian blur 软化

**MuseTalk 现状**：
- 用 BiSeNet parser 生成 mask（11 类：face/skin/lips/eyes/...）
- 已经支持 `parsing_mode='lips_only'` / `'lips_outer_only'`
- mask 边界有 `lips_blend_dilation` + `blend_mask_blur_ratio` 控制

**借鉴优先级**：🟢 低（架构差异大）

**理由**：
- LatentSync 是 diffusion inpainting，**必须** 知道 inpaint 区域
- MuseTalk 是 encoder-decoder 单步生成，BiSeNet mask 只用于 paste-back 混合，不需要 inpaint 区域
- 强行抄 LatentSync 的方案会让 MuseTalk 多一次 landmark→mask 的开销，得不偿失

**唯一可借鉴**：`9cb71d4` "Pad all_dynamic_masks for inference-skip batches"——MuseTalk 的 `_write_result_frames` 在 `passthrough` 段也会写帧，确保 mask 尺寸一致。这个已经在做了。

---

## §E. Unit tests（MuseTalk 严重缺）

**LatentSync 现状**（`tests/` 目录，4 个文件，~85K 总）：
- `test_codeformer_integration.py` (43.7K)
- `test_yaw_filter.py` (25.3K)
- `test_post_processing.py` (13.9K)
- `test_temporal_continuity.py` (3.5K)

**MuseTalk 现状**：
- 只有 `test_ffmpeg.py`（基础 smoke test）
- `quality_optimizations_2026_06.md` 章节 8.4 的 8 个必测用例**只描述了 fixture，没写代码**
- 之前用户选择"只加 fixture 规格，不写测试代码"——这个选择可以重新审视

**借鉴优先级**：🟢 高（任何新策略 §A/§B/§C 都应该有对应测试）

**建议**：
- 优先级排序：先补 `test_segment_consistency.py`（覆盖 `_enforce_segment_consistency` 的 6 个核心场景）→ `test_temporal_blend.py`（覆盖 output_temporal_blend + CodeFormer EMA）→ `test_codeformer.py`（覆盖 fallback 触发）
- 框架用 pytest（LatentSync 也用）
- 框架代码可以从 LatentSync `tests/` 抄过来——API 字段名一致

**代码改动预估**：~400 行（含抄过来的脚手架）
**风险**：低，纯增量

---

## §F. DeepCache 2x speedup（MuseTalk UNet 不适用）

**LatentSync 做法**（`37afbc7` "Re-enable DeepCache by default for ~2x speed"）：
- DeepCache 缓存 UNet 中间特征，每 3 帧跳 2 帧 forward
- 节省 ~66% UNet 推理时间，整体 ~2x speedup
- 代价：边缘细节略糊，嘴部动作 still OK

**MuseTalk 现状**：
- UNet 是**单步**生成（不是扩散）
- 单步 UNet 没有"中间 timestep 可以跳"的概念
- DeepCache 优化不适用

**借鉴优先级**：⚪ 不适用

**但 LatentSync 的 perf 思路可借鉴**：
- `b462062` "Cache per-frame mouth masks between inference loop and restore_video"——和 MuseTalk 的 `62e225a` BiSeNet cache 思路完全一致
- 两者都把"在两段独立代码里都算一遍"的 mask/cache 共享掉

**建议**：Mu**seTalk 已经在做这件事了，§A 完成后可以再加一层 landmark EMA cache**——landmark EMA state 在 inference loop 和 restore_video 都需要（如果 §A 实现的话），可以共享

---

## §G. 状态泄漏修复（MuseTalk 没遇到，但要知道模式）

**LatentSync 做法**（`11a8ba9` "Reset AlignRestore p_bias per inference batch" / `169fb86` "Fix prev_yaw leak"）：
- 单例组件（AlignRestore 缓存 p_bias；yaw 估计器缓存 prev_yaw）在**多次 inference batch**之间会泄漏状态
- 修复：在 batch 边界主动 reset 单例

**MuseTalk 现状**：
- `MuseTalkApiRuntime` 是单例，但**没有 mutable 单例状态**（detector/parser 都是 stateless 包装）
- 没看到类似 bug

**借鉴优先级**：🟢 低（MuseTalk 架构上没这个问题）

**但**——如果未来 §A 引入 LandmarkEMA state，需要在 `synthesize` 开头显式 reset：
```python
landmark_ema_state = LandmarkEMA()  # per-request, not singleton
```

---

## §H. 双方**已有**且一致的策略（值得点名的 cross-validation）

| 策略 | MuseTalk commit | LatentSync commit | 一致性 |
|---|---|---|---|
| 时序 smoothing 在切脸处断开 | `ce7b684` (CodeFormer EMA + track_id) | `e56727d` (general face switch break) | ✓ 思路一致 |
| Temporal EMA 限制到 inpaint/嘴部 region | `5374be1` (output_temporal_blend smoothstep on mouth delta) | `80406f4` (EMA only in inpaint region) | ✓ 思路一致 |
| 嘴部 color match | `67c2f7c` (skin mask) | `cd01aef` (feather band via GPU max-pool) | 🔀 MuseTalk 用静态 mask，LatentSync 用动态 dilation |
| 跨帧 mask cache | `62e225a` (BiSeNet cache) | `b462062` (mouth mask cache) | ✓ 思路一致 |
| CodeFormer 配 higher fidelity + EMA | `3e92ae1` / `90db126` | `08cb35f` (Harden CodeFormer) | ✓ 一致 |
| 收紧 yaw 阈值 45°→30° | (未实现) | `b28a8ae` | ❌ MuseTalk 缺 |
| Unit tests for post-process | (缺) | `a01ae25` | ❌ MuseTalk 缺 |

---

## §I. MuseTalk 优先借鉴清单（按 ROI）

按"价值/成本"排序，给下一轮工作排个序：

| 优先级 | 借鉴项 | 价值 | 成本 | 备注 |
|---|---|---|---|---|
| 🟢 1 | §E Unit tests | 高（解锁后续所有测试） | 400 行 | 纯增量；先做 `test_segment_consistency.py` |
| 🟢 2 | §B Yaw 过滤 | 高（侧脸/转头是高频坏例） | 150 行 + landmark 校准 | DWPose 已经能算，免费 |
| 🟡 3 | §A Landmark EMA | 中（mask 边界稳定性） | 30 行 | §B 落地后再做，先用 §E 测出当前 landmark 抖动程度 |
| 🟡 4 | §C CodeFormer tier | 中（短剧场景） | 30 行 | 客户报告退化严重时再做 |
| ⚪ 5 | §D Dynamic inpaint mask | 不适用 | - | 架构差异，不做 |
| ⚪ 6 | §F DeepCache | 不适用 | - | UNet 单步不适用 |
| ⚪ 7 | §G 状态泄漏 reset | 暂时不适用 | - | §A 落地时顺手加 |

---

## §J. 时间线对照（用于回顾）

```
                  MuseTalk                          LatentSync
2026-05-29                                        (start of window)
2026-06-04  5374be1 output_temporal_blend          8b5fdbd Cross-fade at yaw boundary
              strengthen                          54569d7 Add CodeFormer
2026-06-05  ...                                   3e92ae1 EMA temporal smoothing
                                                    dfdb390 Dynamic mask U-shape
2026-06-08  67c2f7c skin mask for color match     d3a4625 CodeFormer tier-1/2/3
              7b8c49d PNG writes fast-path         b462062 Cache mouth masks
2026-06-09  62e225a BiSeNet cache                 0cde0a5 Multi-signal yaw
              e6f22c6 output_temporal_blend         f6b3652 Tighten yaw filter
                                                    37afbc7 DeepCache re-enable
2026-06-10  90db126 CodeFormer only adjacent EMA  08cb35f Harden CodeFormer
              3e92ae1 CodeFormer EMA + fidelity
2026-06-11  aa8a653 segment time-window merge
              e396ebf majority vote
2026-06-12  4b4987a HeyGen-like segmentation p1
              ce7b684 CodeFormer EMA track-aware
```

**观察**：
- LatentSync 在 6/4~6/5 集中爆发（CodeFormer + yaw 集中落地）
- MuseTalk 在 6/9~6/12 集中爆发（CodeFormer + segment consistency 集中落地）
- 双方在 6/8~6/9 都有 "EMA temporal smoothing" 主题（MuseTalk `e6f22c6` / LatentSync `82e07f6`）——**独立实现，思路一致**
- 双方都在做"CodeFormer 集成 + 稳定化"——MuseTalk 落后约 4 天，但走的是更保守的 fidelity=0.85 路线

---

## 附录：项目差异

MuseTalk 和 LatentSync 是同一个团队（dundun）维护的姊妹项目，API 字段名有意保持一致。架构差异：

| 维度 | MuseTalk | LatentSync |
|---|---|---|
| 模型 | UNet encoder-decoder，**单步**生成 | UNet diffusion，**多步**去噪 |
| 训练目标 | per-frame lipsync + perceptual/GAN/sync loss | multi-step diffusion + syncnet |
| 推理速度 | 30fps+ 实时（单步） | 1-3fps（多步） |
| 嘴部 mask | BiSeNet parser（11 类语义） | landmark 动态 + BiSeNet 混合 |
| 适用场景 | 实时直播、avatar | 高质量离线、影视 |

→ 策略可以互相借鉴，但**实现细节需要适配**：MuseTalk 没有 scheduler/num_inference_steps，所以"扩散专属"参数（cfg/eta）不适用；LatentSync 关心 timestep，所以会有"在 t=0 时"类限定，MuseTalk 直接是最终结果。

---

## §K. 追加：2026-06-13 ~ 2026-06-16 LatentSync 后续两周窗口

更新时间：2026-06-16
窗口外追加：上次的 §J 时间线停到 6/12，6/15~6/16 又有一波"参数微调 + 边界修复"提交，逐一对照如下。

### K.1 切脸/场景切连续性重置（`33bc708` 2026-06-16）

**LatentSync 做法**：在主循环里加 `_source_frame_scene_cut_score(prev, curr)`（BGR 直方图距离 + luma 差），超过阈值时在 `__call__` 入口主动 `reset_p_bias()` + 清空 `prev_yaw` / `prev_motion_state` / `prev_temporal_*` 等 carry state；**不** skip 帧，只重置状态。

**MuseTalk 现状**：
- `ce7b684` CodeFormer EMA 已经按 `track_id` 切换处 reset（`api.py:4726-4736`）
- `segment_consistency_track_aware` 在 hard-cut 处递增 `track_id`（`musetalk/utils/segment_consistency.py:272-285`）
- 也就是**切脸→重置 EMA chain 的路径已经覆盖**——`track_id` 一变 → EMA 跳过 mix → 下一帧重起 chain

**借鉴优先级**：🟢 **不需要新代码**。补一个 test 验证 "hard cut → track_id 切换 → EMA 不混" 已经够了（落到 §E 的 `test_temporal_blend.py`）。

### K.2 默认调参：`min_merged_lipsync_seconds` 和 `mouth_temporal_stabilization_strength`（`7b6fe3a` → `64c65bd` 2026-06-15）

**LatentSync 最终值**（连续两轮微调）：
| 参数 | 起点 | `7b6fe3a` | `64c65bd` 最终 | 原因 |
|---|---|---|---|---|
| `min_merged_lipsync_seconds` | 1.5 | 0.6 | **0.4** | 1.5s 太激进，0.5-1.0s 正常短句/短镜头被错杀；0.4s 留下 flicker 防护栏 |
| `mouth_temporal_stabilization_strength` | 0.15 | — | **0.10** | 0.15 太重，把开口帧平均回去让语音看着"欠发音"；0.10 是平衡点 |
| `mouth_audio_motion_min_scale` | 0.75 | — | 0.85 | 高能语音帧需要保留更多 current-frame motion |
| `mouth_audio_motion_max_scale` | 1.20 | — | 1.35 | 同上 |
| `side_face_episode_pre/post_pad` | 3 | 2 | **1** | 缩短过渡区间，减少"边界不必要切" |
| `yaw_warn_threshold_ratio` | 0.75 | — | 0.80 | 24° 警告带（30°×0.80）减少轻度转头被过度 skip |

**MuseTalk 对照**（**有等价旋钮的两项**）：
- ✅ `min_merged_lipsync_seconds`：默认 1.5 → **0.4**（已应用：`api.py:334-348`）
- ✅ `mouth_temporal_stabilization_strength`：默认 0.08 → **0.10**（已应用：`api.py:362-368`）
- ⚪ `mouth_audio_motion_min/max_scale`：MuseTalk 没有等价参数（嘴部动态通过 `mouth_openness` 链路表达）
- ⚪ `side_face_episode_pre/post_pad` / `yaw_warn_threshold_ratio`：MuseTalk 还没有 yaw 过滤实现（§B 未做），等 §B 落地再考虑

**借鉴优先级**：🟢 **已完成**。两项 Pydantic 默认值已改，文档说明已加。

### K.3 嘴部 mask 钳到 fixed mask 边界（`67e6422` 2026-06-15）

**LatentSync 做法**：`generate_dynamic_mouth_mask(..., fixed_keep_mask=...)` 在末尾做 `torch.maximum(dynamic_keep, fixed_keep)`，防止大张嘴/大笑时 landmark 推导的 mask 溢出下半脸。

**MuseTalk 现状**：
- 用 BiSeNet parser 直接出 `lips_only` / `lips_outer_only` mask（`musetalk/utils/blending.py:226-241`）
- 已有 `lips_blend_dilation`（默认 2 px）做边界外扩
- BiSeNet parser 输出本来就在 trained-on 的下半脸范围内，**溢出风险远低于 landmark 推导**

**借鉴优先级**：🟢 低。架构差异，强行抄会让 MuseTalk 多一次 parser 跑批，得不偿失。

### K.4 `restore_video` 用 `source_frame` 修 IndexError（`9f07a72` 2026-06-15）

**LatentSync 做法**：audio > video 长度时，`loop_video` 返回的 `output_index` 超出 `video_frames` 长度；用 `video_frames[index]` 直接寻址会 IndexError。修复：返回 `source_indices`，`restore_video` 内部用 `source_indices[index]` 寻址 + 钳位。

**MuseTalk 现状**：
- 写帧主循环里写的是 `source_index = self._source_index_for_output(output_index, frame_count)`（`api.py:3685`）
- 然后 `original_frame = frames[source_index].copy()`（`api.py:3686`）

也就是**MuseTalk 早就在用 `source_index` 寻址**（`_source_index_for_output` 内部钳位到 `[0, frame_count-1]`），没有 LatentSync 那个 bug。

**借鉴优先级**：⚪ **不需要**。MuseTalk 架构上没这个 bug。

### K.5 audio sync offset + source affine cache + adaptive quality fallback（`8b05034` 2026-06-15）

**LatentSync 做法**（大组合提交）：
1. `audio2feature.feature2chunks` 加 `offset_seconds` 参数（chunk 生成时就用偏移后的 vid_idx）
2. `loop_video` 改成返回 `source_indices`，避免复制 `video_frames` 数组（音频>视频时长场景）
3. `restore_video` 用 `source_indices` 寻址 + 复用预计算 `dynamic_masks`
4. 加自适应 quality fallback：mouth sharpness ratio + mouth-region diff + identity sim + yaw + audio scale + temporal delta 合成一个 `[0,1]` 质量分，低于阈值回退到 source

**MuseTalk 对照**：
- ⚪ (1) audio offset：**MuseTalk 已有** `audio_sync_offset_seconds`（`api.py:657`），在 `_audio_feature_index_for_output`（`api.py:2971-2984`）做后置寻址；语义上等价于 chunk 时偏移（前提是 audio 覆盖整个 video，没有 loop 截断）
- ⚪ (2)(3) source affine cache：MuseTalk 单步生成没有 per-frame affine 循环（`_encode_latents` 一次性 encode 所有帧 latents），不适用
- ⚪ (4) adaptive quality fallback：MuseTalk 已有 `quality_gate_enabled: bool = False`（默认关闭，opt-in）+ 多项 `quality_*` 旋钮（`api.py:375-403`）。`8b05034` 的合成质量分是更精细的版本，**可以等 MuseTalk 客户实际报退化再升级**（不阻塞）

**借鉴优先级**：🟢 低。功能上 MuseTalk 已经覆盖；如果未来要追平 LatentSync 的 composite score，可以加 `_compute_composite_quality_score` 静态方法，但 ROI 不高。

### K.6 LatentSync 这两周的隐含教训

值得 MuseTalk 记一笔的元规律：

1. **"放松默认"是连续两轮的微调过程**，不是一个数字直接跳到位。`7b6fe3a` 0.6→`64c65bd` 0.4，差 3.5 小时。说明：**调默认是 empirical，每次只挪一小步 + 留灰度让客户可调**。MuseTalk 这次跳 1.5→0.4 步子有点大，建议**先观察 1~2 周**客户端表现，必要时回滚到 0.6。
2. **"切脸 reset" 在 MuseTalk 架构上天然通过 `track_id` 实现了**——`track_id` 是公共的状态总线，新人加任何"per-frame EMA / carry state"时只要遵守 "track_id 变 → reset"，就不用单独搞 `scene_cut_break_*` 旋钮。
3. **`source_frame` vs `video_frames[index]` 的 bug 模式值得记入 memory**：audio > video 长度时 output_index ≠ source_index。MuseTalk 的 `_source_index_for_output` 早处理了，但任何新增的"per-output 写帧"循环都要复用这个 helper。

### K.7 MuseTalk 本轮借鉴落地清单

| 落地项 | 状态 | 改动 |
|---|---|---|
| `min_merged_lipsync_seconds` 默认 1.5 → 0.4 | ✅ | `api.py:334-348` |
| `mouth_temporal_stabilization_strength` 默认 0.08 → 0.10 | ✅ | `api.py:362-368` |
| 跨项目 lessons 文档追加 §K | ✅ | `docs/cross_project_lessons_latentsync_2026_06.md` §K.1~K.7 |
| 给 `min_merged_lipsync_seconds=0.4` 留 1~2 周观察窗 | 🟡 待观察 | 客户端 `passthrough_reasons.too_short` 计数 |
| §E Unit tests for hard-cut → EMA reset | ⏳ 未开始 | 落到 `tests/test_temporal_blend.py` |
| §B Yaw 过滤 + `side_face_*_pad` / `yaw_warn_*` 调参 | ⏳ 未开始 | §I 优先级 🟢 2 |

**LatentSync 这两周的提交里没有能直接搬过来但 MuseTalk 缺的核心策略**。剩下的可借鉴项都是 §I 里已经在跟进的（§A landmark EMA、§B yaw 过滤、§E unit tests）。
