# HeyGen-like Lip-sync Segmentation TD

更新日期：2026-06-10

面向读者：负责 lip-sync / dubbing / video retime 流水线的工程师或 AI coding agent。

本文是一份可迁移的技术设计文档。目标是把我们从商用 HeyGen 样片中反推出来的“稳定片段送模型、疑难片段跳过”的策略，移植到任意 MuseTalk、KeySync、Wav2Lip、Sync Labs、fal lip-sync 或自研 lip-sync 流水线中。

文档不依赖某个项目里的片段编号、函数名或 UI 名称。实现时请把下文里的概念映射到你们自己的 manifest、timeline、clip、task queue 或 media asset 命名。

**相关文档**：
- 近期（2026-06-05 ~ 2026-06-12）50 个 commit 的质量提升与性能优化总结：[quality_optimizations_2026_06.md](./quality_optimizations_2026_06.md)，其中 §2"段一致性切分"是本文档的实操落地记录

## 1. 背景

我们在自建 lip-sync 链路中遇到过几类典型坏例：

- 多角色切换时，把上一位角色的嘴部或眼部贴到下一位角色脸上。
- 模型输出出现固定矩形/椭圆 mask、果冻感、尾帧鬼影。
- 人脸过小、侧脸、遮挡、镜头切换时，嘴部区域重绘不稳定。
- 把 VAD/ASR 检测到的整段说话都送模型，导致模型处理范围过大，错误累积。

对比多个 HeyGen 商用输出后，我们观察到：HeyGen 并不是“检测到说话就整段强重绘”。它更像是先筛选稳定可处理窗口，再对疑难区域跳过或弱处理。它宁可少处理一些片段，也避免明显串脸、mask 跑偏和尾帧污染。

因此，自建链路的目标不是最大化处理秒数，而是最大化“安全可处理秒数”：

```text
稳定单一目标 + 嘴部清晰 + 人脸足够大 + track 连续 => 送 lip-sync 模型
换人/硬切/多人重叠/小脸/遮挡/漂移/过短 => 跳过，保留原视频
同一稳定角色的短静音/短闭嘴间隔 => 合并进同一个送模窗口，减少拼接割裂
```

## 2. 总体策略

策略分两层：

1. 基础召回层：尽可能找出所有可能需要 lip-sync 的候选时间段。
2. 风险过滤层：在候选段内部，只保留高置信、稳定、单目标的嘴动 island；其他区域 passthrough。

不要让 VAD/ASR 单独决定最终送模窗口。音频只说明“有人说话”，不说明“画面里的目标是否适合重绘”。

推荐最终输出三类时间段：

| 类型 | 是否送模型 | 用途 |
| --- | --- | --- |
| 可处理 lip-sync 段 | 是 | 稳定单角色，送 MuseTalk/KeySync 等模型 |
| 风险跳过段 | 否 | 有人脸或可能有人说话，但风险高，保留原片 |
| 普通 passthrough 段 | 否 | 无脸、无人说话、纯背景、转场等 |

## 3. 使用的图像/音频工具

推荐工具组合如下，可替换为等价实现。

### 视频和音频基础工具

- `ffmpeg`: 裁剪视频片段、抽帧、重采样、合并最终输出。
- `ffprobe`: 获取视频时长、fps、分辨率、音频流信息。
- OpenCV: 读帧、计算 ROI 像素差异、颜色直方图、生成 contact sheet、可视化 bbox。
- matplotlib 或任意绘图库: 生成 timeline compare 图。

### 端侧视觉模型

- MediaPipe Face Landmarker 或同等人脸 landmark 模型。
- 需要输出：
  - 人脸 bbox；
  - 嘴部关键点；
  - 眼睛/鼻子关键点用于判断正脸程度；
  - 多人脸检测结果。

### 音频/文本检测

- VAD 可以用 RMS 能量、Silero VAD、Whisper VAD 或 ASR 时间戳。
- ASR 可辅助判断语音段，但不能直接作为最终送模窗口。

### 对比 HeyGen 或其他商用品质时的诊断工具

- 原片 vs 商用输出逐帧对齐。
- 嘴部 ROI 像素差异：用于粗略发现可能被重绘的位置。
- 嘴部 landmark 几何差异：比较原片和商用输出的 mouth open ratio / mouth shape，减少转码噪声干扰。
- contact sheet：把关键帧按时间拼成一张图，人工确认是否跨角色、是否存在 mask 问题。
- timeline 图：横轴时间，标出候选段、送模段、跳过段、商用疑似处理段。

## 4. 输入数据契约

流水线需要准备两类中间数据。

### 4.1 基础候选段

每个候选段至少包含：

```json
{
  "start": 12.34,
  "end": 16.78,
  "duration": 4.44,
  "has_speech": true,
  "has_face": true,
  "candidate_for_lipsync": true
}
```

字段名可以不同，但必须能表达：

- 时间范围；
- 是否有语音；
- 是否有人脸；
- 是否初步认为可能需要 lip-sync。

### 4.2 抽样帧人脸记录

建议以 4-8 fps 抽样。我们当前偏向 6 fps。

每个抽样帧至少包含：

```json
{
  "timestamp": 12.50,
  "frame_width": 1080,
  "frame_height": 1920,
  "scene_cut": false,
  "faces": [
    {
      "track_id": 7,
      "bbox": [420, 300, 780, 760],
      "is_front": true,
      "is_talking": true,
      "mouth_open_ratio": 0.08,
      "mouth_change": 0.04,
      "face_area_ratio": 0.08
    }
  ]
}
```

字段名可替换，但需要表达：

- 当前帧是否硬切；
- 每张脸的稳定身份 track；
- bbox；
- 是否正脸/半正脸；
- 嘴是否在动；
- 人脸占画面面积；
- 嘴部变化幅度。

## 5. 核心判断逻辑

### 5.1 人脸 tracking

为每张脸维护 `track_id` 或等价身份编号。建议匹配依据：

- bbox IoU；
- bbox 中心位移；
- bbox 尺寸变化；
- 人脸 crop 的颜色/外观直方图；
- 如果有条件，可加入 face embedding。

最关键的是：硬切镜头后必须重置 tracker。

推荐做法：

```text
对每个抽样帧计算低分辨率 HSV 直方图。
scene_cut_score = 1 - histogram_intersection(previous_frame_hist, current_frame_hist)
如果 scene_cut_score >= 0.55:
  清空上一帧 tracks
  清空上一帧 mouth state
  当前帧重新分配新的 track_id
```

原因：男女角色在相似构图下切镜头时，bbox 可能高度重叠，肤色/发色直方图也接近；如果不检测硬切，很容易把两个人误认为同一个 track，导致一个送模片段跨角色。

### 5.2 嘴动判断

建议使用嘴部 landmark 计算：

```text
mouth_open_ratio = vertical_mouth_open / mouth_width
mouth_change = abs(current_mouth_open_ratio - previous_mouth_open_ratio_for_same_track)
```

一个帧级 talking face 推荐满足：

```text
is_front == true
mouth_change >= 0.035
mouth_open_ratio >= mouth_open_threshold * 0.5
```

推荐初始阈值：

```text
mouth_open_threshold = 0.055
mouth_change_threshold = 0.035
```

注意：这些阈值不是最终策略，只是“候选召回”。后面还要做稳定性过滤。

### 5.3 talking island

在每个基础候选段内部，遍历抽样帧：

1. 每帧只接受恰好一个 talking face。
2. 如果 0 个 talking face，跳过该帧。
3. 如果多个 talking face，认为目标不唯一，跳过该帧。
4. 连续且同一 track 的 talking samples 合并为一个 island。
5. island 中间允许一个很小 gap，建议不超过 `0.30-0.35s`。

注意：基础候选段内如果出现多位角色，不要整段送模型。应该拆成多个单角色 island。

### 5.4 island 稳定性过滤

一个 island 必须满足：

```text
duration >= 0.45s
talking_sample_count >= 3
face_area / frame_area >= 0.006
bbox 中心漂移 <= 1.25 * median_face_width
bbox 尺寸抖动 <= 1.9x
仅有一个 talking track
```

这些阈值是起点，不是绝对值。不同视频分辨率、采样 fps、人脸检测器会影响阈值。

建议额外增强：

- mouth ROI 不贴边；
- 平均正脸分足够；
- 眼睛/鼻子 landmark 不大幅抖动；
- 嘴部 ROI 清晰度足够；
- face_count 不剧烈变化；
- 角色 face embedding 不能突然变化。

### 5.5 同角色上下文合并

不要把每个嘴动 island 都独立裁剪。这样会造成模型输出和原视频频繁拼接，观感割裂。

如果两个 lip-sync island 满足：

```text
同一 track
中间 gap <= 2.5s
gap 内目标持续可见
gap 内没有其他 talking face
gap 内 bbox 稳定
gap 内没有硬切镜头
```

则可以把中间静音/闭嘴区域一起合并成一个更长送模片段。

典型例子：

```text
0.0-1.5s 女主说话
1.5-3.5s 女主看着对方但不说话
3.5-5.0s 女主继续说话
=> 合并为 0.0-5.0s 一个送模片段
```

反例：

```text
0.0-1.5s 女主说话
1.5-3.5s 镜头切换
3.5-5.0s 男主说话
=> 切成两个片段，中间 passthrough 或 skip
```

### 5.6 短前摇/尾巴吸收

同一稳定角色的短前摇或尾巴也可以吸收进送模段，建议上限 `0.6s`。

用途：

- 避免嘴部区域在一句话刚结束时从模型输出突然切回原片。
- 让模型获得一点上下文，减少边界跳变。

吸收条件：

```text
edge_duration <= 0.6s
相邻 lip-sync 段为同一 track
edge 内目标持续可见
edge 内没有其他 talking face
bbox 稳定
没有硬切
```

### 5.7 短孤立片段跳过

合并同角色上下文之后，如果某个准备送模段仍然很短，建议跳过。

推荐阈值：

```text
如果 merged_lipsync_duration < 1.5s:
  不送模型，保留原视频
```

原因：

- 1 秒左右的孤立短句送模型价值低；
- 裁剪、模型处理、再拼接容易产生割裂；
- 商用链路在类似短孤立片段上经常选择 passthrough 或弱处理。

例外：如果短嘴动 island 能和同一稳定角色的上下文合并到超过阈值，则可以送模型。

## 6. 输出策略

最终每个时间段建议输出：

```json
{
  "start": 12.34,
  "end": 16.78,
  "action": "lipsync | passthrough | skip_risky",
  "reason": "stable_single_speaker | no_face | no_speech | low_confidence | too_short | speaker_switch | hard_cut | small_face | face_drift | multi_speaker_overlap",
  "speaker_track": 7
}
```

推荐 reason 语义：

| reason | 含义 | 现状 |
| --- | --- | --- |
| stable_single_speaker | 稳定单目标，可送模型 | 默认（无显式 reason 计入） |
| no_face | 无人脸 | 现有 `filtered_reason="motion_outlier"` 等之外的 bbox=None 帧 |
| no_speech | 无语音或无有效嘴动 | 现有 audio speech gate（不直接写 reason，provenance 已是 passthrough） |
| low_confidence | 置信不足，不送模型 | 预留位，下一轮实现 |
| too_short | 合并后仍过短 | **新增**（min_merged_seconds 触发） |
| speaker_switch | 前后角色切换，不合并 | **新增**（track-aware 拒绝合并） |
| hard_cut | 硬切镜头边界，不跨越 | **新增**（hard cut detector 拒绝合并） |
| small_face | 人脸过小 | 现有 `face_too_small` |
| face_drift | bbox 漂移过大 | 现有 motion_outlier / fast_motion 近似 |
| multi_speaker_overlap | 多人同时说话或目标不唯一 | 预留位，下一轮实现（当前 pipeline 已在身份匹配阶段过滤非目标人脸） |

api.py 当前用两套并行字典报告：
- `quality_fallback_reasons` = 渲染质量门控（mouth_laplacian / mouth_drift_mse 等 6 个 key）
- `passthrough_reasons` = 切段/合并决策（hard_cut / speaker_switch / too_short 等 10 个 key，见 api.py:3689-3700）

两套字典不重叠：`quality_fallback_reasons` 是"渲染坏了 → 回退到 source frame"，
`passthrough_reasons` 是"切段/合并决策把整段降级到 passthrough"。

## 7. 裁剪和合并

裁剪规则：

1. 每次重跑前，清理旧的 clip、frame preview、reference face 等生成文件。
2. 按最终时间段裁出视频和音频。
3. 只有 `action=lipsync` 的片段送模型。
4. `passthrough` 和 `skip_risky` 使用原视频片段。
5. 模型返回后，按原时间轴拼回。

注意：

- 不要让旧 clip 文件残留在目录中，否则 agent 或人工验收可能误读旧结果。
- 片段 ID 只是调试便利，不应作为业务逻辑依据。
- 合并最终视频时要保持原 fps、分辨率、SAR/DAR、音频采样率一致，避免画面被拉伸。

## 8. 诊断和验收方法

### 8.1 Timeline

生成一张横向时间轴图，至少包含：

- 原始候选段；
- 最终送模段；
- 风险跳过段；
- 硬切位置；
- 多人/小脸/漂移原因。

如果有商用输出对比，可以额外叠加：

- 商用疑似处理窗口；
- retime 映射曲线；
- 嘴部几何变化窗口。

### 8.2 Contact Sheet

对每个高风险片段生成 contact sheet：

- 每 0.3-0.5s 抽一帧；
- 画出 bbox、track id、talking 状态；
- 标出硬切帧；
- 肉眼确认是否跨角色。

Contact sheet 是发现跨男女角色、旧 clip 残留、mask 固定区域的最快方法。

### 8.3 原片 vs 商用输出对比

如果要对齐 HeyGen 或其他商用策略，建议分两步：

1. 像素 ROI 差异：快速找可能处理过的区域，但容易受转码和字幕影响。
2. landmark 几何差异：比较嘴部 open ratio、嘴角宽度、上下唇距离，更适合估算“嘴型是否真的变了”。

不要只用整帧像素差异判断商用处理范围。字幕、压缩、亮度、锐化都会制造误报。

### 8.4 必测用例

实现后至少构造以下测试。每个用例标注了 fixture 描述（视频长什么样、抽样位置、目标 reason 计数），下一轮按此生成 `tests/fixtures/`。

| # | 场景 | fixture 描述 | 目标 reason 计数（api.py） | 关键字段 |
| --- | --- | --- | --- | --- |
| 1 | 同一角色 5 秒稳定镜头，中间有 2 秒不说话 | 25fps / 5s 视频：0-1.5s 说话、1.5-3.5s 张嘴不动、3.5-5.0s 继续说话，bbox 连续 | `passthrough_reasons.hard_cut == 0`，`speaker_switch == 0`，`too_short == 0` | `merge_window=1.0` 应合并为单段 |
| 2 | 前半女主说话，后半男主说话 | 25fps / 4s 视频：0-2s 女主 bbox 稳定可见，2-2.1s 切镜头（构图相近），2.1-4s 男主 bbox 稳定可见 | `passthrough_reasons.speaker_switch >= 1` 或 `hard_cut >= 1`（取决于人脸相似度） | `track_aware=True` + `hard_cut_enabled=True` 应拒绝合并 |
| 3 | 硬切后脸框位置相近，也必须重置 track | 25fps / 4s 视频：0-2s 镜头 A（构图 A），2-4s 镜头 B（构图 B 几乎相同但内容不同） | `passthrough_reasons.hard_cut >= 1` | `hard_cut_threshold=0.65` 应触发 |
| 4 | 小脸候选应跳过 | 25fps / 3s 视频：人脸 bbox 面积 < 0.5% 帧面积 | `filtered_small_face_frames >= N`（已有 counter） | `min_face_area_ratio=0.005` |
| 5 | 多人同时说话应跳过 | 25fps / 3s 视频：同帧 2 个 talking face | `filtered_mouth_diff_frames >= 1`（已有 counter） | 当前 pipeline 已在身份匹配阶段过滤非目标 |
| 6 | 低于 1.5 秒的短孤立片段应跳过 | 25fps / 2s 视频：1s 说话 + 1s 张嘴不动，bbox 稳定可见 | `passthrough_reasons.too_short >= 30`（30fps 下 45 帧） | `min_merged_lipsync_seconds=1.5` |
| 7 | 短 island 合并上下文后保留 | 25fps / 5s 视频：0-1.2s 说话 → 1.2-3.0s 张嘴不动 → 3.0-4.2s 说话 | `passthrough_reasons.too_short == 0`，合并段时长 ≈ 4.2s | `merge_window=2.5` 应跨静音合并 |
| 8 | 重跑拆分时旧 clips 被清理 | 任何上面用例重跑一次 | `results/api/outputs/{job_id}/` 目录仅保留新产物 | 服务端 job_id 隔离 |

opt-out 一致性测试（验证不引入回归）：

| # | 场景 | fixture | 期望 |
| --- | --- | --- | --- |
| 9 | 默认行为（无新字段） | 任何用例用未设置新字段的 payload | `passthrough_reasons.hard_cut/speaker_switch/too_short` 在无对应场景时全部为 0 |
| 10 | 全部 opt-out | `hard_cut_enabled=False` + `track_aware=False` + `min_merged_lipsync_seconds=0` | 与 5374be1 commit 当前行为完全一致；3 个新 key 全 0 |

## 9. 建议默认参数

```text
# 通用抽象参数（与具体实现无关）
frame_sample_fps = 6.0
mouth_open_threshold = 0.055
mouth_change_threshold = 0.035
max_talking_gap_inside_island = 0.30-0.35s
min_island_duration = 0.45s
min_talking_samples = 3
max_bbox_center_drift_face_widths = 1.25
max_bbox_size_jitter_ratio = 1.9
same_actor_edge_context_seconds = 0.6

# api.py 当前实现的实际默认值（已对齐到代码，2026-06-12）
# 字段名直接对应 LipSyncRequest
min_face_area_ratio = 0.005                    # lipsync_min_face_area_ratio
same_actor_bridge_seconds = 1.0                 # segment_consistency_merge_window_seconds
min_merged_lipsync_seconds = 1.5                # min_merged_lipsync_seconds
scene_cut_histogram_threshold = 0.65            # segment_consistency_hard_cut_distance_threshold（注意比抽象值 0.55 更保守，留出亮度突变安全垫）
track_aware_merge = True                        # segment_consistency_track_aware
segment_consistency_passthrough_ratio = 0.5     # majority vote 阈值
```

注意：
- 抽象参数中的 `min_face_area_ratio = 0.006` 是通用建议；api.py 实际用 0.005（256×256 crop 下更稳）。
- 抽象参数中的 `same_actor_bridge_seconds = 2.5` 是 doc 设计目标；api.py 实际用 1.0（同窗口同时承担 island 内 talking gap 与同 actor bridge，speak continuity 优先）。如需拆分两个旋钮，参考章节 12 后续工作。
- 抽象参数中的 `scene_cut_histogram_threshold = 0.55` 是 doc 设计起点；api.py 实际用 0.65（更保守，避免把补光/闪光误判为硬切）。

调参原则：

- 错误重绘多：提高 face area、降低 bbox drift 容忍、提高 min duration；检查 `passthrough_reasons.hard_cut / speaker_switch` 是否非零。
- 漏掉稳定说话多：降低 min island duration 或 mouth_change threshold，但不要放弃稳定性过滤；检查 `passthrough_reasons.too_short` 是否过高的同时 majority vote 是否在合理通过。
- 频繁切碎：提高 same actor bridge，但必须保证不跨硬切、不跨角色；新加的 `segment_consistency_hard_cut_enabled=False` 可临时绕过做对照实验。
- 串脸：加强 hard cut、face embedding、speaker switch 检测；`passthrough_reasons.speaker_switch` 计数应大于 0。

## 10. Agent 实现提示词

可以把下面这段交给另一个 AI coding agent：

```text
请在现有 lip-sync 流水线中实现一个 HeyGen-like segmentation strategy。

目标：
不要把 VAD/ASR 检测到的说话长段整段送 lip-sync 模型。先在候选段内部筛出稳定、可见、单一目标的嘴动 island；再把同一稳定角色镜头内的短静音/闭嘴间隔合并回送模片段；疑难区域 passthrough，保留原视频。

实现要点：
1. 抽样帧做人脸 landmark 和 tracking。
2. 用整帧 HSV 直方图检测硬切；硬切后重置 tracker。
3. 用 mouth_open_ratio 和 mouth_change 召回 talking face。
4. 每帧只接受恰好一个 talking face；多人同时 talking 不要硬选。
5. 把连续同 track talking samples 合并为 island。
6. island 必须满足最小时长、最少采样、人脸面积、bbox 中心稳定、bbox 尺寸稳定。
7. 同一 track 的相邻 island 如果中间目标持续可见、没有其他人说话、bbox 稳定、没有硬切，则合并为一个送模段。
8. 同一角色的短前摇/尾巴可以吸收进送模段，上限约 0.6s。
9. 合并后仍短于 1.5s 的孤立送模段降级为 passthrough/skip。
10. 每次重跑前清理旧 clips，避免过期片段误导验收。
11. 输出每段 action 和 reason，方便调试。

请先写测试：
1. 同一角色跨静音合并。
2. 换人切开。
3. 硬切重置 track。
4. 小脸跳过。
5. 多人重叠跳过。
6. 短孤立跳过。
7. 短 island 合并上下文后保留。
8. 重跑不残留旧 clip。

验收：
生成 timeline 图和 contact sheet；确认最终送模段不跨角色、不跨硬切；最终只有 action=lipsync 的片段进入模型，其余使用原片 passthrough。
```

## 11. reason ↔ 代码映射表（api.py 实现）

按本文档章节 6 的 reason 分类，与 api.py 当前实现位置一一对应。便于"按文档排查"和"按代码排查"互查。

| reason | api.py 字段 / 函数 | 现状 |
| --- | --- | --- |
| `no_face` | `targets[i].get("bbox") is None` | 现有（无显式计数，provenance="passthrough" 隐含） |
| `no_speech` | audio speech gate（`_audio_activity_mask`） | 现有（无显式计数） |
| `face_too_small` | `_filter_lipsync_targets`（api.py:2771） | 现有，写 `filtered_reason="face_too_small"` |
| `short_target_segment` | `_filter_lipsync_targets`（api.py:2771） | 现有，写 `filtered_reason="short_target_segment"` |
| `motion_outlier` | `_filter_motion_targets`（api.py:2565） | 现有 |
| `fast_motion` | `_filter_fast_motion_targets`（api.py:2614） | 现有 |
| `mouth_diff_break` | `_filter_mouth_diff_targets`（api.py:2708） | 现有 |
| `speaker_switch` | `_enforce_segment_consistency` Step 2 | **新增（本次实现）** |
| `hard_cut` | `_enforce_segment_consistency` Step 1.5 + Step 2 | **新增（本次实现）** |
| `too_short` | `_enforce_segment_consistency` Step 3 末尾 | **新增（本次实现）** |
| `low_confidence` | （无） | 预留位，下一轮实现 quality-gate 复用 |
| `multi_speaker_overlap` | （无） | 预留位，下一轮实现 |
| `face_drift` | `_filter_motion_targets` / `_smooth_target_bboxes` 近似 | 现有，间接覆盖 |
| `stable_single_speaker` | （隐式） | 默认（无显式 reason 计入） |

报告字段映射（api.py 2026-06-12 现状）：

- `quality_fallback_reasons`（api.py:3679-3686）= 渲染质量门控，6 个 key
- `passthrough_reasons`（api.py:3689-3700）= 切段/合并决策，10 个 key
- `frame_provenance` = 每帧 passthrough/generated/quality_fallback/blend_error 4 选 1

## 12. 版本维护

后续如果策略迭代，建议只更新这几个区域：

- 默认参数；
- reason 分类；
- hard cut / face embedding / mouth visibility 规则；
- 必测用例；
- agent 实现提示词。

这样不同流水线可以保持策略语义一致，即使内部变量名、UI 名、文件结构完全不同。
