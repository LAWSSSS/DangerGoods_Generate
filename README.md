# 废钢背景异物合成数据生成工具

## 逐件尺寸和 Prompt（2026-10-14）

已建立桌面六视图目录中 **38 张参考图**的独立配置，背景目录共 **73 张**。当前 `config.yaml` 启用 `use_object_catalog: true`，从 `profiles/object_profiles.json` 读取单件尺寸/材质/遮挡与 prompt，六视图路径为 `/Users/liuzhaocheng/Desktop/六视图`。

每件以 `S01`–`S38` 标识，不再只凭大类套用同一尺寸，也不混入同类其它型号的图。完整参数和逐件提示词在 [逐件手册](profiles/物品尺寸与Prompt逐件手册.md)，可编辑原始工作值在 [配置种子](profiles/object_profile_seeds.json)，独立文本在 `profiles/prompts/`。

```bash
python object_catalog.py --list
python object_catalog.py --item S15
python object_catalog.py --item S26 --output output/item_prompts/S26.txt
# 下面会发送真实模型请求，仅在明确要生成且素材就绪时运行
python generate_aliyun.py --objects S15 --total 1
```

工作尺寸包含参考范围和证据状态；多数不是实测。UCP210 可按型号表使用尺寸，其它未确认型号以可调整工作值开始。背景水平宽度 2.4m 沿用旧程序假设，未标定；画面占比同时考虑投影与遮挡，不把可见片段当成完整物体。

已分别配置 DN25 小阀与 DN100/DN200/DN350 工业阀，立式/卧式储罐，三种千斤顶，空气与弹簧减震器。塑料、不锈钢、铝和橡胶不会统一处理为厚铁锈。新的专属 prompt 写在可写的 `profiles/` 中，保留原来只读的 `prompts/` 目录。

修改 `profiles/object_profile_seeds.json` 后执行 `python tools/build_object_profiles.py` 可重建参数与文本。设 `use_object_catalog: false` 则恢复旧的按类别入口。当前只完成配置与离线验证，未调用生图/VLM API。

---

本项目基于阿里云百炼平台（Wan2.7-Image 模型 + Qwen-VLM），用于批量生成"废钢车斗背景 + 微小异物"的高仿真合成图片，主要面向废钢安检、异物检测、目标识别模型训练的数据增强场景。

核心特点：
- **VLM 闭环优化**：生成后自动用视觉大模型评估异物大小/融合度，不合格自动优化 Prompt 重试
- **Prompt 自动学习**：自动区分保存"好Prompt"和"坏Prompt"，积累生成经验
- **智能尺寸计算**：根据背景图分辨率自动计算异物最优像素尺寸
- **加权采样机制**：支持通过文件名前缀控制异物出现概率
- **多层Prompt增强**：自动添加位置、遮挡、做旧、光影、融合等18项约束

---

## 目录

1. [工作原理](#1-工作原理)
2. [目录结构说明](#2-目录结构说明)
3. [环境安装](#3-环境安装)
4. [配置说明](#4-配置说明)
5. [输入数据准备](#5-输入数据准备)
6. [Prompt 模板配置](#6-prompt-模板配置)
7. [使用方法](#7-使用方法)
8. [输出详解](#8-输出详解)
9. [VLM 闭环优化机制](#9-vlm-闭环优化机制)
10. [Prompt 自动学习机制](#10-prompt-自动学习机制)
11. [BBox 标注输出](#11-bbox-标注输出)
12. [重要细节说明](#12-重要细节说明)
13. [常见问题](#13-常见问题)
14. [调优建议](#14-调优建议)

---

## 1. 工作原理

### 整体流程

```
输入准备
    ↓
随机选择背景图 + 加权选择异物图
    ↓
┌─ VLM 分析输入图片（可选，默认开启）
│   ├─ 分析异物：类别、颜色、形状、材质、纹理、关键特征
│   └─ 分析背景：光照、阴影、堆叠方式、缝隙位置、整体色调
    ↓
生成初始 Prompt（结合分析结果 + 模板 + 18项融合约束）
    ↓
┌─ VLM 优化循环（最多 N 轮，默认3轮）
│   ├─ 调用 Wan2.7-Image 生成图片
│   ├─ VLM 评估：尺寸判断/置信度/融合质量
│   ├─ 合格 → 跳出循环
│   └─ 不合格 → 根据反馈优化 Prompt，重试
    ↓
最终正式生成
    ↓
保存：最终图片 + 中间轮次图片 + 元数据 + Prompt记录
```

### 核心技术点

1. **不是传统图生图**：不是简单的贴图，而是让 AI 根据参考图和 Prompt 重新绘制，异物会自然变形、做旧、被遮挡
2. **异物参考图的作用**：作为视觉参考传给模型，告诉模型"要加的东西长什么样"，**不会直接粘贴到背景上**
3. **背景图完全保留**：Prompt 中强制要求"保持原始背景完全不变"，理论上除了异物区域其他像素不应改变
4. **尺寸硬约束**：根据背景分辨率动态计算目标尺寸（4K图：0.8%-1.8% 画面宽度，2K图：1.2%-2.5%，小图：1.5%-3%）

---

## 2. 目录结构说明

```text
auto_generate1/
├── generate_aliyun.py          # 主程序（主入口，用于批量生成）
├── generate.py                 # 与 generate_aliyun.py 完全相同的副本（内容一致，任取其一即可）
├── test_new_api.py             # 独立脚本：测试/验证 API Key 是否有效
├── config.yaml                 # 默认配置文件（主程序实际加载此文件）
├── config_bailian.yaml         # 百炼(Token Plan)备用配置，内置 api_key
├── requirements.txt            # Python依赖
├── debug_response.json         # 调试用：API返回异常时自动保存完整响应
│
├── input/                      # 输入目录
│   ├── backgrounds/            # 废钢车斗背景图（所有jpg直接放这里，不用分子文件夹）
│   └── objects/                # 异物参考图（所有jpg直接放这里，通过文件名识别类别）
│
├── prompts/                    # Prompt配置目录（自动生成，可手动修改）
│   ├── prompt_templates.json   # 每个类别的基础Prompt模板
│   └── negative_prompt.txt     # 负向提示词（禁止生成的内容）
│
└── output/                     # 输出目录（自动创建）
    ├── images/                 # VLM优化过程中每一轮的中间生成图片
    ├── final_images/           # 最终合格的生成图片（训练用这个目录）
    ├── metadata/               # 每张图的完整元数据JSON
    ├── failed/                 # 生成失败的任务记录（含错误信息）
    └── analysis/               # Prompt学习数据：好Prompt/坏Prompt/统计信息
```

---

## 3. 环境安装

### 3.1 Python 版本要求
- Python 3.8+

### 3.2 安装依赖
```bash
pip install -r requirements.txt
```

依赖包含：
- `requests>=2.32.3`：HTTP请求
- `PyYAML>=6.0.2`：配置文件解析
- `Pillow>=10.4.0`：图像处理

### 3.3 API Key 配置
两种方式二选一：

**方式一：环境变量（推荐）**
```bash
# Windows
set DASHSCOPE_API_KEY=你的API密钥

# Linux/Mac
export DASHSCOPE_API_KEY=你的API密钥
```

**方式二：配置文件**
直接在 `config.yaml` 中填写 `api_key` 字段。

---

## 4. 配置说明

所有配置在 `config.yaml` 中修改；密钥、厂商地址和同套模型选择也写在 `config_bailian.yaml`。

### 4.1 模型选择
生图 / VLM 默认由路由函数选择；`route` 暂时留空时，依次读取 `config.yaml`、`config_bailian.yaml` 的 `model` / `vlm_model`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `route` | `""` | 路由函数入口，暂时留空 |
| `model` | `wan2.7-image` | 生图模型。可选：`wan2.7-image`、`wan2.7-image-pro`、`doubao-seedream-5.0-lite`、`doubao-seedream-5.0-pro` |
| `vlm_model` | `qwen3.8-flash` | VLM 模型。可选：`qwen3.8-flash`、`qwen3.8-max`、`doubao-seed-2.1-pro` |

两个 YAML 里都可以写这三项；同时存在时以 `config.yaml` 为准。生图和 VLM 可以分属阿里和火山，程序按模型名绑定协议和 Key。

### 4.2 图像生成参数
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `size` | `""` | 输出尺寸，空表示与背景图一致 |
| `output_format` | `png` | 输出格式：png/jpg/webp |
| `watermark` | `false` | 是否添加水印（**必须设为false**） |
| `max_images` | `1` | 每次生成返回几张图 |

### 4.3 融合参数（重要！调优关键）
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `object_resize_scale` | `0.018` | 异物基础缩放比例（1.8%画面宽度） |
| `object_canvas_scale` | `1.0` | 参考图画布扩展比例（**当前版本未实际使用**） |
| `object_opacity` | `0.5` | 参考图透明度（**当前版本未实际使用，靠Prompt控制**） |
| `object_anchor` | `center` | 异物对齐方式（**当前版本未实际使用**） |

> ⚠️ **注意**：`object_resize_scale` 是基础值，实际尺寸会根据背景分辨率动态计算，VLM优化时还会动态调整。

### 4.4 VLM优化参数
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_vlm_optimization` | `true` | 是否开启VLM闭环优化 |
| `max_optimization_rounds` | `3` | 最大优化轮数（每轮都要调用API，会增加耗时和成本） |
| `size_tolerance` | `very_small` | 尺寸容忍度 |

### 4.5 运行参数
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `timeout_seconds` | `300` | API超时时间（秒） |
| `sleep_seconds` | `1.0` | 每张图生成间隔（秒），防止限流 |
| `total_num` | `100` | 默认生成数量 |
| `seed` | `null` | 随机种子，固定后可复现结果 |

### 4.6 目录配置
一般不需要修改，除非你要自定义路径：
- `backgrounds_dir`：背景图目录
- `objects_dir`：异物参考图目录
- `prompt_templates_path`：Prompt模板路径
- `negative_prompt_path`：负向提示词路径
- 各类输出目录...

---

## 5. 输入数据准备

### 5.1 背景图准备

**位置**：`input/backgrounds/`

**要求**：
- 格式支持：`.jpg` / `.jpeg` / `.png` / `.webp`
- 直接把所有背景图放到这个目录下，**不需要分子文件夹**
- 建议分辨率：1920x1080 以上，越高清生成效果越好
- 内容：真实的废钢车斗照片，包含各种废钢堆叠场景

**命名**：无特殊要求，任意文件名均可。

### 5.2 异物参考图准备

**位置**：`input/objects/`

**⚠️ 重要：不需要按类别分子文件夹！所有异物图直接放在这个目录下即可！**

#### 类别识别规则
程序通过**文件名包含关键词**自动识别异物类别，支持的类别（对应prompt_templates.json中的key）：

| 类别关键词 | 中文含义 |
|-----------|----------|
| `fire_extinguisher` / `灭火器` | 灭火器 |
| `gas_cylinder` / `气瓶` | 气瓶 |
| `shell` | 炮弹 |
| `lpg_tank` / `储罐` / `液化气罐` | 液化气罐/储气罐 |
| `accumulator` / `蓄能器` | 蓄能器 |
| `hydraulic_cylinder` / `液压缸` / `油缸` / `气缸` | 液压缸/油缸/气缸 |
| `shock_absorber` / `减震器` | 减震器 |
| `sealed_container` / `密闭` | 密闭容器 |
| `tire` / `轮胎` | 轮胎 |
| `roller` / `滚筒` | 滚筒 |
| `wire_rope_drum` / `卷筒` | 钢丝绳卷筒 |
| `axle` / `桥` | 车桥/轴类 |
| `reducer` / `减速机` | 减速机 |
| `track_roller` / `支重轮` | 支重轮 |
| `drive_sprocket` / `驱动轮` | 驱动轮 |
| `wheel_end` / `轮边` | 轮边总成 |
| `guard_block` / `防护墩` | 防护墩 |
| `千斤顶` | 千斤顶 |
| `过滤器` / `分离过滤器` | 过滤器 |
| `漆罐` / `自喷漆` / `杀虫剂` | 各类罐体 |
| `集油器` | 集油器 |
| `轴承座` | 轴承座 |
| `球阀` / `阀门` | 阀门 |
| `胶罐` / `密封胶` | 密封胶罐 |

> 文件名只要包含上述关键词之一就会被识别为对应类别，例如：
> - `灭火器.jpg` → fire_extinguisher
> - `液压缸1.jpg` → hydraulic_cylinder
> - `aDN350 球阀.jpg` → （会匹配不到，需要在prompt_templates中添加对应类别，或改名为包含关键词）

#### 加权采样机制（重要！）

**文件名以小写字母 `a` 开头的图片，被选中的概率权重为 0.1**，其他正常图片权重为 1.0。

用途：
- 对于样本数量多但不想让它出现太频繁的异物，在文件名前加 `a`
- 例如：`aDN350 球阀.jpg`、`a密封胶罐.jpg`、`a轴承座.jpg` 这些低概率异物
- 效果：如果有10张正常图 + 3张a开头图，总权重=10*1 + 3*0.1=10.3，a开头图被选中概率约为 0.1/10.3 ≈ 1%

### 5.3 参考图质量建议
- 背景图：清晰、光线均匀、废钢纹理丰富
- 异物图：背景干净最好，异物主体完整
- 异物图不需要裁剪得特别精确，模型会自己识别主体

---

## 6. Prompt 模板配置

首次运行程序会自动在 `prompts/` 目录生成默认模板，你可以手动修改优化。

### 6.1 prompt_templates.json
结构：`{"类别名": ["模板1", "模板2", ...]}`

每个类别可以配置多个基础Prompt，生成时随机选择一个，然后程序会自动在后面追加18项增强约束。

**默认模板示例**：
```json
{
  "fire_extinguisher": [
    "在废钢堆的缝隙中嵌入一个极小的红色灭火器，被废钢遮挡只露一小部分，表面有灰尘划痕锈迹。",
    "在废钢堆深处放入一个极小的灭火器，被遮挡不显眼，保持灭火器的典型形状，表面旧化带锈。"
  ]
}
```

### 6.2 negative_prompt.txt
负向提示词，告诉模型不要生成什么。默认内容已经比较完善，一般不需要修改。

默认禁止项：
- 改变原始背景
- 卡通风格
- 干净崭新的异物
- 异物悬浮
- 异物过大（不超过画面5%）
- 改变异物形状
- 亮面/光滑/反光物体
- 异物90%以下遮挡
- 人工放置感

---

## 7. 使用方法

### 7.1 基础命令
```bash
python generate_aliyun.py
```
使用config.yaml中的默认配置生成。

> 💡 说明：`generate.py` 与 `generate_aliyun.py` 内容完全一致，二者任选其一作为入口均可。`test_new_api.py` 是一个独立小脚本，仅用于验证 API Key 是否有效（不参与批量生成）。

### 7.2 常用命令示例

```bash
# 生成100张图
python generate_aliyun.py --total 100

# 指定随机种子（可复现）
python generate_aliyun.py --total 50 --seed 42

# 关闭VLM优化（速度快3倍，成本低，但效果差一些）
python generate_aliyun.py --total 100 --no-vlm

# 设置VLM最大优化轮数为5（更严格，但更慢更贵）
python generate_aliyun.py --total 100 --max-rounds 5

# 随机选择背景图（默认是按顺序遍历）
python generate_aliyun.py --total 200 --random

# 指定只使用某些背景图
python generate_aliyun.py --total 10 --backgrounds "2026_07_13_36_2_zhongfei1_100_109.jpg" "2026_07_13_36_2_zhongfei1_100_110.jpg"

# 指定只使用某些异物图（可写文件名或路径；多张时按权重从中选取）
python generate_aliyun.py --total 10 --objects "灭火器_01.jpg" "气瓶_02.jpg"

# 同时指定背景和异物
python generate_aliyun.py --total 5 --backgrounds "bg_a.jpg" --objects "密封胶罐.jpg"

# 仅分析已有的Prompt统计数据，不生成图片
python generate_aliyun.py --analyze-only
```

### 7.3 命令行参数完整列表

| 参数 | 说明 |
|------|------|
| `--total N` | 本次生成N张图，覆盖配置文件中的total_num |
| `--seed N` | 指定随机种子 |
| `--no-vlm` | 关闭VLM闭环优化，直接生成 |
| `--max-rounds N` | VLM最大优化轮数 |
| `--backgrounds 文件1 文件2 ...` | 指定只使用列出的背景图（文件名或路径） |
| `--objects 文件1 文件2 ...` | 指定只使用列出的异物图（文件名或路径；多张时加权选取） |
| `--random` | 随机选择背景图（默认顺序遍历） |
| `--analyze-only` | 不生成，只分析并打印Prompt统计 |

---

## 8. 输出详解

### 8.1 什么被保存了？什么没被保存？

✅ **被永久保存的内容**：
1. **最终生成图片**：`output/final_images/` （训练集用这个）
2. **VLM优化过程的中间图片**：`output/images/` （每一轮重试的图都保存，方便调试）
3. **完整元数据JSON**：`output/metadata/` （包含所有信息）
4. **失败记录**：`output/failed/` （错误信息）
5. **Prompt学习记录**：`output/analysis/` （好Prompt、坏Prompt、统计）
6. **原始背景图路径**：元数据中记录了使用的背景图绝对路径
7. **原始异物图路径**：元数据中记录了使用的异物参考图绝对路径
8. **所有Prompt版本**：基础Prompt、优化后Prompt、最终完整Prompt、负向Prompt
9. **VLM分析结果**：对输入图的分析JSON、每一轮的评估结果
10. **VLM优化历史**：每一轮的Prompt、评估结果、改进建议
11. **API完整返回结果**：metadata中保存了API返回的完整JSON
12. **BBox坐标**：像素坐标 + 归一化坐标
13. **质量评估字段**：size_judgment、confidence、quality_score、fusion_quality

❌ **没有被保存的内容**：
1. **原始背景图本身**：不备份，只保存路径
2. **原始异物参考图本身**：不备份，只保存路径
3. **上传到API的图片base64**：不保存
4. **下载的临时图片**：最终保存到磁盘，内存中不缓存

### 8.2 最终图片命名规则
```
{序号}_{背景图文件名}_{异物类别}_{时间戳}.png
```
示例：`000001_2026_07_13_36_2_zhongfei1_100_109_fire_extinguisher_20260716T083000Z.png`

### 8.3 元数据JSON字段说明
每个metadata JSON包含以下字段：

```javascript
{
  "image_id": "唯一图片ID",
  "index": 序号,
  "background_image": "背景图绝对路径",
  "object_image": "异物图绝对路径",
  "object_class": "异物类别名",
  "base_prompt": "从模板中选的基础Prompt",
  "final_prompt": "与 full_prompt 相同，即最终发给API的完整Prompt",
  "full_prompt": "最终发给API的完整Prompt（含18项约束+负向）",
  "negative_prompt": "负向提示词全文",
  "model": "使用的图像模型",
  "generated_images": ["最终图片路径列表"],
  "vlm_enabled": true/false,
  "vlm_used": true/false,
  "vlm_success": true/false,  // VLM是否判定合格
  "vlm_analysis": { /* VLM对输入图的分析结果 */ },
  "optimization_rounds": 优化轮次,
  "optimization_history": [ /* 每一轮的详细记录 */ ],
  "bbox": [x1, y1, x2, y2],  // VLM给出的 bounding box
  "quality_status": "pending_review",
  "api_result": { /* API返回的完整响应 */ }
}
```


## 9. VLM 闭环优化机制

这是本项目最核心的功能，默认开启。

### 9.1 优化流程
1. **输入分析**：VLM同时看异物图和背景图，分析物体特征和环境特征
2. **初始Prompt生成**：基于分析结果，自动生成包含位置、遮挡、做旧、光影的详细Prompt
3. **第一轮生成**：调用Wan生成图片
4. **VLM评估**：VLM看生成结果，回答：
   - 异物是 too_large（太大）/ acceptable（合格）/ too_small（太小）？
   - 置信度多少？
   - 融合质量 bad/ok/good？
   - 异物的BBox坐标？
   - 应该怎么改进Prompt？
5. **判断是否重试**：
   - 如果判定 acceptable → 优化结束，进入最终生成
   - 如果判定太大/太小 → 根据VLM建议自动修改Prompt，回到步骤3重试
   - 连续2轮判定太大 → 下一轮强制大幅缩小
   - 连续2轮判定太小 → 下一轮强制略微放大
6. **达到最大轮数**：不管合格与否都停止，用最后一轮的Prompt做最终生成

### 9.2 尺寸调整策略
| VLM判定 | 下一轮尺寸调整 |
|---------|---------------|
| too_large | 缩小到基础值的 40%-60%，要求更深地嵌入 |
| too_small | 放大到基础值的 130%-180%，确保形态完整 |
| acceptable | 保持尺寸，强调边缘遮挡 |

### 9.3 成本与耗时
- 每轮优化需要：1次图像生成API + 1次VLM API
- 默认3轮优化 = 最多4次图像生成 + 4次VLM调用
- 关闭VLM（`--no-vlm`）= 1次图像生成，速度快3-4倍，成本低70%以上
- 建议：先小批量测试VLM效果，大批量生成时如果效果稳定可以考虑关闭VLM提速

---

## 10. Prompt 自动学习机制

程序会自动积累生成经验，保存到 `output/analysis/` 目录。

### 10.1 判定标准
每次VLM评估后，自动判定这次的Prompt是"好"还是"坏"：
- ✅ **好Prompt**：size_judgment == acceptable **且** confidence >= 0.5 **且** quality_score >= 0.5
- ❌ **坏Prompt**：其他所有情况

### 10.2 保存的文件
| 文件 | 内容 |
|------|------|
| `good_prompts.json` | 所有好Prompt的完整记录，包含Prompt全文、类别、BBox、评分、时间戳 |
| `bad_prompts.json` | 所有坏Prompt的完整记录，包含失败原因 |
| `prompt_stats.json` | 统计信息：总数、好/坏数量、最后更新时间 |

### 10.3 记录字段
每条Prompt记录包含：
- 完整的Prompt文本（不是摘要，是全文）
- 对应的图片ID
- 异物类别
- 尺寸判定结果
- 置信度、质量分、融合质量
- BBox坐标
- 第几轮优化
- 时间戳
- Prompt长度

### 10.4 用途
- 运行结束自动打印最新的3条好Prompt和3条坏Prompt作为参考
- 可以手动从good_prompts中提取效果好的Prompt，更新到prompt_templates.json中
- 可以分析bad_prompts找出共性问题，针对性调整负向提示词
- 运行 `python generate_aliyun.py --analyze-only` 随时查看统计

---

## 11. BBox 标注输出

### 11.1 标注来源
BBox **不是模型生成时输出的**，是VLM在评估生成图片时"看"出来的边框坐标。

### 11.2 标注精度
- VLM给出的是大致边框，精度在像素级别可能有偏差
- 适合用于：弱监督学习、预标注、难例挖掘
- 不适合直接作为高精度Ground Truth（建议人工抽检复核）


---

## 12. 重要细节说明

### 12.1 关于参考图是否被"保存"
- 程序**不会修改、覆盖、删除** `input/` 目录下的任何原始图片
- 程序**不会备份**原始输入图片，只在元数据中记录文件路径
- 发给API时，图片被转为base64编码在请求体中传输，不在本地留缓存
- 生成的新图片与原始输入图片是独立的文件

### 12.2 关于背景图是否被修改
- Prompt中多次强调"保持原始背景完全不变"，但这是Prompt约束，不是像素级保证
- Wan2.7-Image是重绘模型，理论上可能对背景有细微修改
- 如果需要严格像素级不变，需要后处理做mask融合（当前版本未实现）

### 12.3 关于异物形态保持
- Prompt中反复强调"严格保持物体原本的形状、结构、比例和轮廓特征"
- 但AI生成可能会有变形，特别是物体很小时
- VLM评估时没有专门检查形态，只检查大小和融合度
- 如果形态变形严重，需要在Prompt模板中更强调形状特征

### 12.4 SSL证书问题
- 程序内置了SSL验证绕过，专门处理某些网络环境下的证书问题
- 自动重试机制：5xx错误自动重试5次，指数退避
- 如果还是SSL错误，检查网络代理设置

### 12.5 配置文件注意事项
- 主程序（`generate_aliyun.py` / `generate.py`）实际加载的是 **`config.yaml`**，不是 `config_bailian.yaml`
- `config_bailian.yaml` 是**百炼（Token Plan）专用的备份配置**，内含 api_key，仅作为参考/备用
- 如需切换不同环境的配置，直接把对应配置内容复制到 `config.yaml` 即可
- ⚠️ 生产或协作时**不要把含真实 api_key 的文件提交到版本库**

### 12.6 关于中间图片
- `output/images/` 保存了VLM优化每一轮的中间结果
- `output/final_images/` 是优化结束后再重新生成一次的最终结果
- 所以即使VLM第2轮就判定合格，也会多生成一次最终图
- 中间图可以用来对比优化效果，不需要可以定期清理

### 12.7 随机种子
- 指定seed只能保证随机选择的背景图、异物、Prompt模板可复现
- **不能保证AI生成的图片像素级复现**，因为生成API本身有随机性

---

## 13. 常见问题

### Q1: 提示找不到配置文件 / 读取到空的 config
A: 主程序加载的是 `config.yaml`，请确保该文件存在且 `api_key` 已填写。若提示无法识别配置，检查 `config.yaml` 的 YAML 格式是否合法，或把 `config_bailian.yaml` 的内容复制到 `config.yaml`。

### Q2: 提示 `No background images found`
A: 检查 `input/backgrounds/` 目录下是否有jpg/png图片。

### Q3: 提示 `No object images found`
A: 检查 `input/objects/` 目录下是否有图片，以及文件名是否包含能识别的类别关键词。

### Q3.1: 如何只使用某几张异物图？
A: 使用 `--objects`，参数可以是文件名或路径，支持多张；生成时会在指定集合内按权重选取。例如：
```bash
python generate_aliyun.py --total 10 --objects "灭火器_01.jpg" "气瓶_02.jpg"
```
若指定名称一个都匹配不上，会打印 Warning 并回退到全部异物图。

### Q4: 某些异物图片打印 WARNING: 无法识别类别
A: 文件名不包含任何预设关键词，需要：
- 修改文件名，加上对应类别关键词（如加上"灭火器"、"油缸"等）
- 或者在prompt_templates.json中添加新的类别和对应关键词

### Q5: API报错 401 Unauthorized
A: API Key错误或过期，检查配置文件或环境变量中的Key是否正确。

### Q6: API报错 429 Too Many Requests
A: 触发限流，调大config.yaml中的sleep_seconds，降低并发。

### Q7: 生成的异物还是太大
A: 调优方法：
1. 减小config.yaml中的object_resize_scale（比如从0.018改到0.012）
2. 增大max_optimization_rounds到5
3. 在negative_prompt.txt中更强调"极小"、"几乎不可见"
4. 检查是否关闭了VLM（--no-vlm模式尺寸控制差很多）

### Q8: 生成的异物太小看不见
A: 调优方法：
1. 适当增大object_resize_scale
2. 在Prompt模板中强调"保持形态完整，能识别出是xxx"
3. 减少max_optimization_rounds，避免过度优化

### Q9: 异物形状变形严重
A: 在Prompt模板中更详细描述物体的形状特征，比如"红色圆柱形，顶部有黑色压把和保险销"，而不是只说"灭火器"。

### Q10: 异物看起来很假，像P上去的
A: 确保开启VLM优化，VLM会专门评估融合质量；在Prompt模板中增加做旧、生锈、被遮挡的描述。

---

## 14. 调优建议

### 快速批量出图（优先速度）
```yaml
# config.yaml
enable_vlm_optimization: false
sleep_seconds: 0.5
```
```bash
python generate_aliyun.py --total 1000 --no-vlm --random
```

### 高质量出图（优先效果）
```yaml
# config.yaml
enable_vlm_optimization: true
max_optimization_rounds: 5
object_resize_scale: 0.015
sleep_seconds: 1.5
```
```bash
python generate_aliyun.py --total 100 --max-rounds 5
```

### 新增异物类别
1. 在 `input/objects/` 放入对应异物图片，文件名包含类别中文名
2. 运行一次程序，会自动生成默认prompt_templates.json
3. 编辑 `prompts/prompt_templates.json`，添加新类别的Prompt模板
4. 在代码 `PromptEnhancer.get_weathering_descriptions()` 中可以添加该类专属的做旧描述（可选）

### 持续优化Prompt效果
1. 生成一定数量图片后，运行 `python generate_aliyun.py --analyze-only` 查看统计
2. 打开 `output/analysis/good_prompts.json`，复制效果好的Prompt
3. 更新到 `prompts/prompt_templates.json` 中替换掉效果差的模板
4. 定期清理bad_prompts，形成正向循环

---

## 成本估算（参考）

按阿里云百炼当前定价：
- Wan2.7-Image 生成：约 0.12元/张
- Qwen3.7-Plus VLM：约 0.02元/次

| 模式 | 单张成本 | 1000张成本 | 速度 |
|------|---------|-----------|------|
| 无VLM优化 | ~0.12元 | ~120元 | ~10秒/张 |
| VLM 3轮优化 | ~0.12*4 + 0.02*4 = ~0.56元 | ~560 元 | ~40秒/张 |
| VLM 5轮优化 | ~0.12*6 + 0.02*6 = ~0.84元 | ~840元 | ~60秒/张 |

> 实际成本以阿里云官方定价为准，以上仅为估算。

