# MuseTalk 项目上下文（供 AI 助手快速恢复）

> 此文件由 AI 助手生成，用于后续会话快速恢复项目上下文。请勿手动修改关键结构。

---

## 1. 项目概述

**MuseTalk**：腾讯音乐娱乐 Lyra Lab 开源的实时高保真视频口型同步（lip-sync）模型。在 NVIDIA Tesla V100 上可达 30fps+。不是在潜空间中通过 **单步 inpainting** 生成口型，而非扩散模型。

**两个版本**：
- **v1.0**：基础版本，支持 `bbox_shift` 调节嘴型开合
- **v1.5**（推荐）：集成 perceptual loss、GAN loss、sync loss，视觉质量和同步精度大幅提升

**人脸区域**：256 x 256。支持多语言（中/英/日等）。

---

## 2. 技术架构

### 核心模型组件
| 组件 | 来源 | 作用 |
|------|------|------|
| **VAE** (`sd-vae-ft-mse`) | Stable Diffusion | 图像编解码潜空间 |
| **UNet** | SD UNet 改造 | 单步生成，音频通过 cross-attention 融合 |
| **Whisper-tiny** | OpenAI | 音频特征提取（编码器输出 hidden states） |
| **DWPose** | MMPose | 人脸 68 点 landmark 检测 |
| **SFD** | face-alignment | 人脸检测 bounding box |
| **Face Parsing** | BiSeNet | 生成 blending mask，控制融合区域 |
| **SyncNet** | LatentSync | 训练时唇音同步判别 |

### 数据流
```
视频帧 → 人脸检测(SFD) + Landmark(DWPose) → crop 256×256
                              ↓
音频 → Whisper → 50fps 音频特征 → 按视频 fps 采样 → audio prompts
                              ↓
         VAE encode(masked + ref) → UNet(audio cross-attn) → VAE decode
                              ↓
                    Face Parsing Mask → 融合回原图
```

---

## 3. 项目目录结构

```
MuseTalk/
├── api.py                    # FastAPI 生产服务（核心部署入口）
├── app.py                    # Gradio 演示界面（本地/HuggingFace）
├── train.py                  # 训练入口
├── scripts/
│   ├── inference.py          # CLI 离线推理（单视频）
│   ├── realtime_inference.py # CLI 实时推理（Avatar 预计算）
│   └── preprocess.py         # 数据预处理
├── musetalk/
│   ├── models/
│   │   ├── vae.py            # VAE 包装（AutoencoderKL）
│   │   ├── unet.py           # UNet2DConditionModel + PositionalEncoding
│   │   └── syncnet.py        # 唇音同步判别器
│   ├── utils/
│   │   ├── audio_processor.py # Whisper 特征提取
│   │   ├── preprocessing.py   # 人脸检测/landmark/DWPose
│   │   ├── blending.py        # Mask 生成 & 图像融合
│   │   ├── utils.py           # 模型加载、datagen、工具函数
│   │   └── face_parsing/      # 人脸解析模型
│   ├── data/                  # 训练数据加载
│   └── loss/                  # 训练损失函数
├── configs/
│   ├── inference/             # 推理配置（test.yaml, realtime.yaml）
│   └── training/              # 训练配置（stage1/2, preprocess, gpu）
├── models/                    # 模型权重目录（服务器上名称可能不同）
│   ├── musetalkV15/           # v1.5: musetalk.json + unet.pth
│   ├── musetalk/              # v1.0: musetalk.json + pytorch_model.bin
│   ├── sd-vae/ 或 sd-vae-ft-mse/
│   ├── whisper/               # config.json, pytorch_model.bin, preprocessor_config.json
│   ├── dwpose/, face-parse-bisent/, syncnet/
├── results/api/               # API 输入输出
│   ├── inputs/{job_id}/
│   └── outputs/{job_id}/
└── data/                      # 示例音视频
```

---

## 4. 入口点与使用方式

### A. FastAPI 服务 (`api.py`) — 生产部署

**环境变量/命令行参数**：
- `API_HOST/API_PORT`, `FFMPEG_PATH`, `MUSETALK_GPU_ID`
- `MUSETALK_VERSION` (v15/v1), `MUSETALK_VAE_TYPE`
- `MUSETALK_UNET_CONFIG`, `MUSETALK_UNET_MODEL`
- `MUSETALK_WHISPER_DIR`, `MUSETALK_USE_FLOAT16`
- `MUSETALK_FACE_EMBEDDING_BACKEND` (auto/insightface/none)
- `API_MAX_DOWNLOAD_BYTES`, `API_DOWNLOAD_RETRIES`

**API 端点**：
| 端点 | 方法 | 功能 |
|------|------|------|
| `GET /health` | — | 检测器/模型/embedding 加载状态 |
| `POST /api/faces` | — | 上传视频，提取不同人脸身份，返回排序后的人脸裁剪图 URL |
| `POST /api/lipsync` | — | 视频+音频(+可选 avatar) → 口型同步视频 |
| `GET /api/download` | — | 下载生成结果或代理远程 URL |

**`/api/lipsync` 核心行为**：
- 有 `avatar_url`：只修改匹配到的人脸身份
- 无 `avatar_url`：默认修改视频中出现频率最高的人脸
- 无人脸/无目标/人脸太小/静默音频 → **原帧直出**（pass through）
- 支持短间隙填充、bbox 平滑、运动门控、语音门控

### B. Gradio 演示 (`app.py`)
```bash
python app.py --use_float16 --ffmpeg_path ...
```
- 可调参数：`bbox_shift`, `extra_margin`, `parsing_mode`, `left/right_cheek_width`
- 支持首帧调试（debug inpainting）

### C. CLI 离线推理 (`scripts/inference.py`)
```bash
sh inference.sh v1.5 normal
# 或 python -m scripts.inference --inference_config configs/inference/test.yaml ...
```

### D. CLI 实时推理 (`scripts/realtime_inference.py`)
```bash
sh inference.sh v1.5 realtime
```
- **Avatar 预计算**：首次 `preparation=True`，缓存 latents/coords/masks
- 后续同一 Avatar 只需替换音频，达到 30fps+ 实时

---

## 5. API 服务端详细流程 (`api.py` → `synthesize`)

```
1. 下载输入（video, audio, avatar）
2. 读取视频帧 & fps
3. 提取 avatar 人脸描述子（InsightFace embedding + 视觉描述子）
4. 扫描视频帧，聚类人脸身份，确定目标身份
   - 有 avatar：匹配相似度最高的身份
   - 无 avatar：选出现频率最高的身份
5. 逐帧匹配目标 bbox（身份匹配 + landmark 修正 + bbox_shift）
6. 后处理目标序列：
   - _fill_short_target_gaps: 填充短间隙
   - _filter_motion_targets: 过滤大幅度运动
   - _filter_fast_motion_targets: 过滤快速突变
   - _fill_short_target_gaps (continuity): 连续性再填充
   - _filter_lipsync_targets: 过滤小脸、短片段
   - _smooth_target_bboxes: bbox 平滑
7. 加载 MuseTalk 模型（VAE, UNet, Whisper, PE）
8. 提取音频特征（Whisper）
   - 支持 audio_feature_fps / max_audio_feature_fps 独立控制
9. 语音门控（speech gate）：静音帧直出
10. 对目标帧 VAE encode → latents
11. 构建 process_items（output_index → audio_index + latent）
12. 批量推理（UNet + PE + audio cross-attention）
13. 逐帧渲染：VAE decode → resize → Face Parsing Mask → 融合回原图
14. 质量门控：模糊帧 fallback 到原图
15. ffmpeg 合成视频 + 音频
16. 返回结果 URL + 详细统计信息
```

---

## 6. 核心模块职责

| 模块 | 职责 |
|------|------|
| `musetalk/utils/audio_processor.py` | Whisper 音频预处理。注意保留小数 fps（29.97, 23.976） |
| `musetalk/utils/preprocessing.py` | DWPose landmark + SFD 人脸检测。全局初始化 `model` 和 `fa` |
| `musetalk/utils/blending.py` | `get_image_prepare_material()`: mask + blur；`get_image_blending()`: 粘贴融合 |
| `musetalk/utils/face_parsing/` | BiSeNet 人脸解析，生成精细 mask |
| `musetalk/utils/utils.py` | `load_all_model()`: 加载 VAE/UNet/PE；`datagen()`: 推理 batch 生成器 |

---

## 7. 训练流程

**数据预处理**：
```bash
python -m scripts.preprocess --config configs/training/preprocess.yaml
```

**两阶段训练**：
```bash
sh train.sh stage1   # 单帧，大 batch（32），学基础口型
sh train.sh stage2   # 16 帧时序，小 batch（2）+ grad accum（8），学时间一致性
```

---

## 8. 关键行为与约束（来自 AGENTS.md）

- **模型路径不硬编码**：服务器上 VAE 可能是 `sd-vae-ft-mse`，权重可能在 `models/musetalk`。用 `_resolve_*` 动态检测。
- **分数 fps 保留**：29.97、23.976 不得截断为整数。
- **分离两扇 gate**：身份匹配（改谁） vs 语音活动（改不改）。静音帧即使有目标人脸也直出。
- **InsightFace 优先**：身份判断优先用 embedding，失败 fallback 到视觉描述子（hist/DCT/LBP/gradient）。
- **保守填充**：只在确认目标帧之间填充短间隙，大位移/长间隙保持原帧。
- **输出帧数 = 源视频帧数**：音频短则后面原帧保留，音频长不扩展视频。
- **修改后检查**：`python -m py_compile api.py`，`git diff --check`。
- ** surgical changes**：只改必要代码，不重构未损坏的代码，不清理预存死代码（除非用户要求）。

---

## 9. 依赖环境

- Python 3.10, CUDA 11.7/11.8
- PyTorch 2.0.1
- `diffusers==0.30.2`, `transformers==4.39.2`, `accelerate==0.28.0`
- MMLab: `mmcv==2.0.1`, `mmdet==3.1.0`, `mmpose==1.1.0`
- `opencv-python`, `librosa`, `einops`, `gradio`, `fastapi`, `uvicorn`
- FFmpeg（必须可执行）

---

## 10. 最近变更（供快速追踪）

- **api.py**: 新增 `audio_feature_fps` 和 `max_audio_feature_fps` 参数，将音频特征 FPS 与源视频 FPS 解耦。新增 `_resolve_audio_feature_fps` 和 `_audio_feature_index_for_output` 方法。同步偏移分别按音频特征域和输出帧域计算。
