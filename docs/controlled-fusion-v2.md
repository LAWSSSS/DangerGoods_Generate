# 手册驱动的受控融合引擎

版本：程序 `0.3.0`；任务契约 `2.0`；依据手册 `V2.0`；日期 `2026-10-15`。

## 1. 当前交付与适用范围

本分支把《多物品多场景旧化与真实融合_超级详细实施手册》中的工程约束写入可执行 Python 流程。
原始手册逐字保存在 `docs/specifications/多物品多场景旧化与真实融合_超级详细实施手册_v2.0.md`。
历史 `generate.py`、`generate_aliyun.py` 仍是各自独立入口。新流程入口为 `fusion.py` / `scene-fusion`。

### 已实现的确定性操作

- 严格任务/验收 JSON Schema、私有提供者配置、源文件 SHA-256、输入变更检查。
- 多物品目录迁移；逐件真实尺寸状态、部件树、身份正/负约束和材质旧化计划。
- 原背景坐标中的旋转 OBB；参照尺度区间；显式针孔相机矩阵的三维点投影。
- 支点到轮廓/承载区的距离、重心投影与支撑凸包的必要条件检查。
- 每个交叠区域的遮挡图；按局部像素检查循环，允许不同区域有不同前后次序。
- 六类二值蒙版、编辑/保护冲突检查、原前景恢复、最终 PNG 保护区零差分检查。
- 可选且有界的局部 LAB、掩膜归一化高斯、局部噪声、仅向编辑区内部羽化。
- Wan/Seedream 图像与 Doubao/Qwen VLM 的配置接入；预算、原图起点的修复请求、请求摘要和实际输出记录。
- 外部候选导入、最终母版哈希绑定的人工验收、独立实际轮廓的几何指标、实际轮廓标注导出。
- 手册离线工具、E0–E7 消融计划生成、结构化失败根因与返工阶段。

### 需要输入证据或专用适配器的部分

本版本没有自动重建精确三维模型、推断绝对深度、自动生成可靠完整轮廓、运行有限元/刚体力学、计算真实折射，或读取模型内部去噪轨迹。
场景/物体解析目前接受目录和人工/CAD/经审核分割结果；配置中的光照是带来源的假设。
透明物体与软体对象被标记为 `SPECIAL_RENDERER_REQUIRED`。只有外部生成图也可经 `--candidate` 导入，但需要相应物理/标注证据才能验收。
三维模拟、分割和原生 inpaint 的接口可继续扩展；本版本不把它们标作已执行能力。

## 2. 文件树与职责

```text
fusion.py                              从检出目录直接运行
pyproject.toml                         打包/依赖/scene-fusion 命令
config/controlled/
  rusted_S01.draft.json                 已关联用户生锈气瓶参考
  rusted_S20.draft.json                 已关联用户生锈阀门参考
  rusted_S01_S20.draft.json              两件共同场景草案
src/fusion_engine/
  controlled/
    schema.py / job.py                 契约、路径、事实状态与证据
    catalog.py                         逐件目录迁移、消融实验计划
    geometry.py                        OBB、尺度区间、针孔投影、局部DAG
    planning.py                        场景配置/支撑/材质/联合提示词
    masks.py                           六Mask构建、验证、规划预览
    appearance.py                      分材质旧化与局部相机风格匹配
    adapters.py                        能力声明、实际请求、VLM辅助意见
    quality.py                         硬门禁、独立实测轮廓、RLE标注
    workflow.py                        预算、阶段日志、验收、导出
    cli.py / utilities.py              显式操作命令
  providers/                           外部YAML、图像编码、HTTP、VLM
  schemas/                             可供其他语言使用的JSON Schema
  artifacts.py / images.py             原子JSON、哈希、RGB8/PNG
scripts/fusion_utilities.py             手册离线工具入口
docs/controlled-fusion-v2.md             本文件
docs/manual-v2-implementation-matrix.md  手册章节与代码/能力的映射
docs/CHANGELOG-controlled-fusion.md      程序与资料版本说明
output/controlled_fusion/               忽略Git的实际运行记录
work/manual-v2/                         忽略Git的规划和离线检查
```

`ObjectParser` 对应目录/部件树解析，`SceneParser` 对应带证据的场景区域配置，`PlacementSolver` 对应几何和支撑检查，`InteractionSolver` 对应局部遮挡，`MaskBuilder` 对应六蒙版；`GeneratorAdapter` 是类型化协议。精确三维求解器和分割模型可接入这些输入接口。

## 3. 安装、检查与命令

Python 3.11+。本机已有 Anaconda 环境可直接运行。

```bash
python -m pip install -r requirements.txt
python -B fusion.py --help
python -B fusion.py --version
# 可选：在独立环境中安装命令行入口
python -m pip install -e .
scene-fusion --help
```

新入口在运行前不会启动旧程序的 `total_num=100` 批处理。
`schema`、`validate`、`plan`、`ablation-matrix`、`provider-info`、`request-preview`、`migrate-catalog` 不请求模型。
`run --candidate` 离线处理显式导入的候选。`run --generate` 才调用图像模型；VLM 调用受任务预算控制。

### 3.1 当前用户资产的迁移草案

已生成三份草案，分别关联当前两件生锈六视图及原背景，含源文件哈希。
尺寸沿用既有逐件工作值并保留 `estimated`，没有把六视图或旧的 2.4m 跨度假设升级为场景标定。
初始化 OBB 是图像空间排布草案；不是已经找到的支撑位置。
蒙版与支点为空时，离线规划会返回 `INPUT_REQUIRED`；`run` 不会在这一状态付费生成。

```bash
python -B fusion.py validate config/controlled/rusted_S01_S20.draft.json
python -B fusion.py plan config/controlled/rusted_S01_S20.draft.json \
  --output work/manual-v2/my-joint-plan
```

`plan` 产出 `placement.json`、六类全局/逐件 Mask、`prompt.txt` 和 `placement_preview.png`。
绿色框是完整 OBB；橙色是可编辑区；蓝色是原前景。预览不是融合结果。
缺失轮廓的全零文件仅用于表示缺失，不是分割真值；规划 JSON 同时列出缺失项。

### 3.2 从旧目录初始化其他物品

```bash
python -B fusion.py migrate-catalog \
  --catalog profiles/object_profiles.json \
  --background input/backgrounds/2026_08_06_36_1_zhongfei1_95_zhongfei2_5_19.jpg \
  --objects S01 S20 \
  --reference 'S01=/Users/liuzhaocheng/Desktop/生锈六视图/02_生锈CCS气瓶_六视图.png' \
  --reference 'S20=/Users/liuzhaocheng/Desktop/生锈六视图/01_DN100_手轮阀_重度生锈六视图.png' \
  --output work/manual-v2/new-job.json
```

不提供 `--reference` 时，按旧目录逐件选择原六视图。覆盖参考文件时会重新计算哈希。
每件至少补充真实的结构关键部件和材料分区，例如灭火器的橡胶软管/压力表玻璃不能与钢瓶共享铁锈规则。
迁移不会复制旧 Prompt 中的“第一张背景”顺序或可见长度比例；新请求始终把原背景放最后，并记录每张参考对应的物品/视图 ID。

## 4. 三种模式与执行前状态

| 模式 | 用途 | 背景要求 | 最终状态/证据 |
|---|---|---|---|
| A | 概念草案/路线探索 | 不承诺未编辑区零差分 | `CONCEPT_ONLY`，不导出已验收 final |
| B | 有边界的工程融合 | 编辑区外及显式保护像素零差分 | 独立完整/可见轮廓、硬门禁和哈希绑定人工审阅 |
| C | 可追溯严格标注 | B 的要求以及实际审核标注 | 实际轮廓、标注出处、最终 PNG、COCO兼容RLE |

`READY` 只表示输入已具备执行条件；不表示生成结果已经合格。
状态流：`PREPARED → INPUT_REQUIRED`，或 `PREPARED → 生成/导入 → REVIEW_REQUIRED/REJECTED → 人工证据 → ACCEPTED`。
预算耗尽只记录 `budget_exhausted`，不能把未知项或平均分最高的图片升级为 `ACCEPTED`。
流程错误为 `FAILED`；该运行保留审计记录，重新执行产生新运行目录。

退出码：0 成功规划/已接受/概念产物，2 配置错误或输入待补，3 硬失败拒收，4 等待证据/人工验收。

## 5. 事实、估计、假设与证据字段

每个有物理含义的输入使用 `evidence`。例如：

```json
{
  "source": "proposal",
  "confirmed": false,
  "notes": "示例：尚未核对的放置建议",
  "confidence": 0.2
}
```

`confirmed=true` 必须与实际审核行为一致，填写实际 `reviewer`。掩膜/接触等硬几何输入只接受 human/CAD/calibrated_geometry 的审核证据。
来自 VLM/分割器的结果即使置信度高，也需要先对照源图审核；可将审核后的证据来源写为 human，并在 notes 中保留模型来源。
字段不授权程序捏造审核或代填“已确认”。

真实尺寸状态为 `measured / catalog_verified / estimated / unknown`，带单位和来源。
当前迁移的尺寸为 estimated，不能拿来声称已测量物品长度或背景宽度。
全局 `scene.lighting`、`camera_style` 在规划中标为配置假设；局部统计是实际解码图像的观测。

## 6. 尺度、姿态与支撑

### 6.1 坐标规范

原背景 EXIF 方向校正后建立坐标：x向右，y向下。归一化点 `(x/W,y/H)`。
OBB 的 width 乘原图宽，height 乘原图高；`rotation_deg` 是图像平面顺时针角度。
它不是三维 yaw/pitch/roll。`allow_crop=false` 时，旋转后的四角也必须在画布内。

### 6.2 四个尺度等级

- L0：直接指定像素/归一化投影；不宣称物理标定。
- L1：`p_ref × L_obj/L_ref × Z_ref/Z_obj × gamma_obj/gamma_ref`。两个比值都用区间；缺项返回 UNKNOWN，不默认等于1。
- L2：显式提供 K/R/t 和单位一致的三维点，计算针孔投影；不自动重建或校准，也不计算未给出的镜头畸变。
- L3：明确假设、置信来源及像素区间的候选尺度；仍是估计。

```bash
python -B scripts/fusion_utilities.py scale \
  --object-length 1400 --reference-length 1000 --reference-pixels 200 \
  --depth-ratio-range 0.8 1.1 --foreshortening-ratio-range 0.6 0.9 --units mm
python -B scripts/fusion_utilities.py project --camera-json work/my-camera.json
```

`camera-json` 结构：`{"K":[[fx,0,cx],[0,fy,cy],[0,0,1]],"R":[...3x3...],"t":[tx,ty,tz],"points_3d":[...Nx3...],"units":"mm"}`。
R 必须为有效旋转，点必须处于相机前方。程序记录输入假设和输出投影，不生成虚构重建误差。

### 6.3 支撑输入

`scene.support_regions` 包含独立 ID、实际区域 mask、承载状态和说明。
每件 `support_contacts` 引用支撑区域或其他物品，包含接触点和证据。
程序检查支点到物体完整轮廓和承载区域的像素间隙；有三点和重心投影时检查图像平面的凸包。
这是必要条件。摩擦系数、质量、三维法向和变形未知时，不能据此声称力学稳定；`stability_evidence` 必须说明真实依据。

## 7. 六类蒙版：从哪里来、如何处理

所有二值 Mask 为原背景大小、单通道 1/L、仅0和255。白色表示对应语义为真；不得自动缩放或自动阈值化软 alpha。

| 名称 | 含义 | 输入/派生 |
|---|---|---|
| M_obj_full | 物体完整投影的实际形状 | 提供轮廓，不用OBB替代 |
| M_fg_occ | 保持原像素的真实前景 | 显式输入；无前景需明确empty并说明 |
| M_obj_visible | 去除原前景/其他物体遮挡后的物体 | full 减去经审核的遮挡 |
| M_contact | 支撑接触/允许局部阴影交互区域 | 显式输入；不自动画大黑圈 |
| M_edit | 物体可见区+接触区+允许边界带 | 可派生或显式提供；不能漏有效目标 |
| M_protect | 原图恢复区域 | 编辑区外+叠字+原前景+显式保护区 |

支持 PNG 文件，也支持原背景归一化多边形。以下只是数据格式，不代表已有实例审核：

```json
{
  "polygons": [[[0.30,0.40],[0.56,0.37],[0.59,0.48],[0.32,0.51]]],
  "coordinate_space": "normalized_original_background",
  "evidence": {
    "source": "proposal", "confirmed": false,
    "notes": "示例多边形；需换成对应真实轮廓"
  }
}
```

空区域用 `{"empty":true,"evidence":{...}}`，不能省略后让程序猜测。
Mask 路径为 workspace 相对路径，或显式外部只读绝对路径。可以附 `sha256` 锁住文件版本。
`obj_full`、`fg_occ`、`contact` 缺失时，B/C 执行停留在 INPUT_REQUIRED。

### 多物品遮挡

每对交叠物品必须定义局部 `scene.occlusion_edges`，包含 front_id/back_id、交集区域和审核证据。
同一像素不能有 A在B前、B在C前、C在A前。不同位置可以有不同次序，例如交叉细杆；程序按实际区域检查，不对全图强行做一个排序。
未覆盖的交叠像素标 UNKNOWN；当前实现要求显式说明每对交叠，不自动用未经审核的三维深度补足。

## 8. 材质、光照和局部成像

旧化规则按部件区分碳钢、喷漆钢、铸铁、不锈钢、铝、镀锌、铜、黄铜、塑料、橡胶、木、纸、玻璃、布、陶瓷与未知材料。
全局等级0–4只是计划强度，不是某个固定锈蚀面积。
钢件约束脱漆与露钢锈蚀的位置；不锈钢只在有依据的位置茶渍/点蚀；非铁材料不自动长红锈。
暴露历史、接缝、摩擦面、流路和结构损伤权限进入联合 Prompt。材质计划是请求，不是已完成的 PBR 渲染结果。

后处理顺序：先拿到几何/接触合理的候选 → 可见物体局部有界LAB → 局部成像 → 内向羽化 → 原前景/保护像素恢复 → 保存无损PNG → 重新解码验收。

```json
{
  "mode": "bounded_lab",
  "reference_roi": {
    "path": "work/annotation/comparable_steel_roi.png",
    "evidence": {"source":"proposal","confirmed":false,"notes":"需选取同深度、同材质且照明可比的原图区域"}
  },
  "comparable_roi_confirmed": false,
  "max_lab_shift": [12,5,5],
  "strength": 0.5,
  "blur_sigma_px": 0.35,
  "noise_std": 0.4,
  "noise_seed": 20261015,
  "feather_radius_px": 2
}
```

该配置在 comparable_roi_confirmed=false 时拒绝运行 LAB 校正。未经确认不要改 true。
对照区必须非空且在编辑区外。LAB8 统计未做 ICC 色彩管理，不能当作可溯源色差计读数。
不同材料不要求颜色均值一致；大色差先检查材质和光照，不能仅增大校正上限。
高斯和加噪只写可见目标像素，不能通过整图模糊掩盖几何错误。

## 9. 模型接入和实际能力

```bash
python -B fusion.py provider-info --provider-config config_bailian.yaml
python -B fusion.py request-preview config/controlled/rusted_S01_S20.draft.json \
  --provider-config config_bailian.yaml --output work/manual-v2/request-preview.json
```

当前本地 YAML 绑定 `wan2.7-image` 与 `doubao-seed-2.1-pro`；程序读取该文件，不把密钥复制进任务 JSON、日志或 Git。
`model` 与 `vlm_model` 独立绑定，输出图像档位、水印、n、显式 seed 来自该配置。运行预算在任务 JSON 中单独声明。
旧配置的目标可见比例、透明度、total_num 和迭代数不替代新任务的完整 OBB、Mask 和调用预算。
每份 manifest 会注明这项差异。

当前 Wan 使用原生多模态编辑接口，参考在前、原背景最后；原生 bbox_list 每图最多两框。多于两件时显式采用包围它们的区域框，并记录 `bbox_union_used`；逐件 OBB 仍进入 Prompt。
依据：[阿里云Wan编辑说明](https://www.alibabacloud.com/help/en/model-studio/wan-image-edit)、[Wan2.7 API](https://www.alibabacloud.com/help/en/model-studio/wan-image-generation-and-editing-api-reference)，核对日期2026-10-15。
接口区域框约束不是二值 Mask。当前 Wan/Seedream 适配器声明 `native_binary_mask=false`，`depth_control=false`，`normal_control=false`。
request-preview 实际执行图像输入编码和参数构造，只保存脱敏摘要；即使草案未READY也可检查接口映射，execution_allowed会为false，不发送请求。
`route=native_mask` 会报错；`guided_generation_then_restore` 明确生成后在本地恢复背景，仍需接缝和真实感验收。

输出配准有两种显式策略：

- strict：服务输出与原画布尺寸不同即拒绝。
- resize_same_aspect：允许原比例误差≤2.5%的输出用LANCZOS还原到原画布尺寸，记录服务尺寸/最终尺寸/操作。该步骤不证明几何正确。

导入外部候选要求已经同尺寸，不做隐式配准。支持任意模型生成的候选。
新模型协议接入实现 `GeneratorAdapter`，声明真实 capability，并沿用统一计划、蒙版、输出和验收接口；不得用原生mask标识包装仅支持文本/框的模型。

## 10. 执行与预算

输入证据齐全并且 READY 后，显式执行：

```bash
# 离线处理外部已生成候选
python -B fusion.py run work/manual-v2/confirmed-job.json \
  --candidate work/manual-v2/registered-candidate.png

# 实际付费/配额调用；本次实现期间未执行此命令
python -B fusion.py run work/manual-v2/confirmed-job.json \
  --generate --provider-config config_bailian.yaml
```

预算例：`{"max_image_calls":1,"max_vlm_calls":1,"max_candidates":1}`。
调用计数在网络发送前持久化，超时也算一次已尝试调用；不自动重试。服务可能已计费，因此不能把超时当作免费重试依据。
图像返回数量不能超出本任务剩余候选预算。多轮只在 VLM 明确建议 refine 且还有预算时提出修复请求；每轮以原背景和原参考为起点。
VLM accept 不触发最终接受；其 JSON 不合法会记录 UNKNOWN。
不给模型未公开的真实种子、采样器、内部扩写、注意力图或去噪步编造记录。

## 11. 最终验收、实际标注与防止误接收

机器检查最终 PNG 的保护像素，不能以 JPEG 预览作为零差分验收对象。
每件实际生成物体要提供独立完整轮廓和可见轮廓：来自最终母版的人工描绘，或人工审核后的分割/CAD对齐结果。
计划 Mask 不能直接复制后宣称是实际标注；只有确实观察核对与母版一致时，才能这样确认。
强遮挡完整形状无法可靠恢复时记录 UNKNOWN，不能通过缩小完整尺寸让几何误差消失。
裁切对象目前无法在此二维实现中验证画布外完整尺度，几何门禁保持 UNKNOWN。

### 11.1 几何指标

- center_error：实际完整轮廓拟合 OBB 的中心与计划中心距离 / 原图对角线。
- scale_log_error：`abs(log(wg/wt))+abs(log(hg/ht))`，在一致坐标/完整投影上计算。
- rotation_error_deg：图像平面矩形方向误差，独立关键部件方向仍由身份验收核对。
- mask IoU：实际完整/可见轮廓分别对计划轮廓，不能跨语义比较。
- contact_gap：声明支点到实际完整轮廓边界的像素距离，不能当作三维支撑证明。
- foreground/protect：实际可见物体标注不能侵占恢复的原前景及保护像素。

### 11.2 审核流程

每个候选生成 `analysis/candidate-001.review-template.json`，初始所有人工项为 UNKNOWN。
`config/controlled/review.example.json` 提供轮廓字段的完整格式，但其零哈希未绑定任何候选；实际操作应从本次运行生成的模板开始。
复制到工作区，填写真实审核人、记录和逐项结论，补充每件实际轮廓路径及审核证据。
`mask_matches_master_confirmed` 只有实际核对后才能为 true。

```bash
python -B fusion.py review output/controlled_fusion/TASK/RUN \
  --candidate-id candidate-001 --evidence work/my-human-review.json
```

母版 SHA 不匹配、原始输入变化、配置被改、硬门禁失败、未知项未补齐，都不能导出最终 ACCEPTED。
人工意见不能覆盖像素/几何等硬失败。每次审阅独立保存，旧记录不覆盖。
冻结实际审核轮廓后重新计算质量。只有全通过才复制 `outputs/final.png`；已接受结果不可覆盖。
实际标注导出 `annotations/instances.json`、`observed_masks/`，含COCO兼容非压缩RLE、可见bbox、面积、实例ID及扩展amodal字段。
同一样本的同类物品共用category_id；这些ID在单样本内有效，合并数据集时按类别名称建立全局映射。
非压缩counts数组用于数据集标注，依据[官方COCO的annToRLE实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)；它不等同于loadRes要求的压缩预测结果格式。本次未运行COCO SDK导入验证。
全零、未审核、过期或重叠冲突的实际轮廓不能作为有效标注。

### 11.3 人工检查项

identity、support、occlusion、material、lighting、style、object_count、realism 均保留独立 PASS/FAIL/UNKNOWN。
先缩略图检查整体摆放/风格，再100%看接触与杆件连续，200%看关键连接；不能要求原始低清相机不存在的细节。
综合分数不抵消任何硬失败。VLM意见与人工结论分别保存。

## 12. 离线工具与历史参数

```bash
python -B scripts/fusion_utilities.py inspect --image input/backgrounds/SCENE.jpg
python -B scripts/fusion_utilities.py compare-background \
  --original work/original.png --candidate work/master.png --mask work/M_protect.png
python -B scripts/fusion_utilities.py restore-protected \
  --original work/original.png --candidate work/candidate.png --mask work/M_protect.png \
  --output work/restored.png
python -B scripts/fusion_utilities.py composite \
  --original work/original.png --candidate work/candidate.png --mask work/M_edit.png \
  --output work/paste-baseline.png
python -B scripts/fusion_utilities.py grade-case \
  --image work/legacy-second-round.png --output work/legacy-grade.jpg
```

工具均拒绝覆盖已有输出。compare/restore 的白色为保护；composite 白色为编辑。
工具兼容原手册的 --input、--reference-meters/--object-meters、--composite、--protect-mask、--background、--object-layer、--visible-mask 参数名；也可使用上述新名称。
relative-depth 明确定义为 Z_obj/Z_ref，因此换算为公式中的逆比；direction-factor 是 gamma_obj/gamma_ref。显式单值会记录为退化区间，缺失项仍为UNKNOWN。
比较/尺度工具的FAIL退出3、UNKNOWN退出4；默认tolerance=0，非零容差不会宣称严格像素锁。
composite 是粘贴基线，没有物理融合或真实感合格声明。
空保护区输出 UNKNOWN，不会得到空范围的假 PASS。

grade-case 保留先前会话报告的实际历史参数：OpenCV LAB8，L'=1.20L+9；a'=128+.92(a−128)−2.5；b'=128+.89(b−128)−4；裁剪0–255；高斯σ=.55；JPEG quality=87。
它不是默认生产调色。历史原始生成图/脚本若未在本地提供，不能声称本次再次验证了当时文件的哈希；不同OpenCV/Pillow版本和编码选项也可能改变字节。
原手册中的两轮生成和历史哈希一致性是先前记录，本分支实施不冒充重新执行。

## 13. 运行产物、可复现性和故障排查

```text
output/controlled_fusion/TASK/RUN/
  job.snapshot.json
  manifest.json                       状态、版本、输入哈希、候选、预算
  prompts/                            联合提示与实际逐轮修复提示
  masks/                              规划六Mask、逐件六Mask
  analysis/placement.json             几何/材质/支撑/缺失证据
  analysis/placement_preview.png      标为规划预览
  analysis/input_inventory.json       实际输入文件哈希
  candidates/                        导入候选或原始服务字节/请求摘要
  outputs/candidate-001.png           本地处理后的无损候选
  analysis/candidate-001.quality.json
  analysis/reviews/UUID/              不覆盖的人工记录、实际轮廓与复算质量
  outputs/final.png                   只有ACCEPTED才存在
  annotations/                       只有实际轮廓通过验收才导出
  logs/events.jsonl / budget.json
```

| 问题/代码 | 首先检查 | 返工方式 |
|---|---|---|
| INPUT_REQUIRED / MISSING_MASK | 是否只有框，没有真实轮廓 | 补齐实际区域及来源，重做计划 |
| SCALE_PLAN_CONFLICT | 尺度区间与完整OBB是否一致 | 调整参照证据或完整投影，不能缩小可见片段冒充全长 |
| SUPPORT_GAP | 声明接触点是否真在轮廓/承载区 | 重新摆放、改支点和接触区 |
| LOCAL_OCCLUSION_CYCLE | 同一区域是否有环 | 按交叉位置拆区域并审核 |
| UNRESOLVED_OVERLAP | 两件交叠是否未定义前后 | 补局部遮挡边，不自动猜深度 |
| OBJECT_PROTECT_CONFLICT | 对象可见部分压到叠字/保护区 | 移动物体或修正错误蒙版 |
| 有锈仍像贴图 | 身份/接触/前景/局部光度分别看 | 按根因返回S05/S06/S07/S09/S10，不能靠整图模糊 |
| 模型返回尺寸不一致 | output_registration策略 | strict拒绝；允许比例配准时显式配置并记录 |
| native_mask报错 | provider capability | 换真实支持的适配器，或明确本地恢复路线 |
| VLM JSON失败 | 辅助审阅记录UNKNOWN | 保留候选和预算，不编造分数或默认接受 |
| 人工报告哈希不匹配 | 是否修改/重新编码候选 | 对当前母版重做审核，不能编辑manifest绕过 |
| final不存在 | 是否仍REVIEW_REQUIRED/REJECTED | 补实际证据或修复失败项；不自动挑最高分 |

## 14. 扩展与后续工程边界

1. 自动分割器：输出带来源、模型版本、置信、原始坐标和明确amodal/visible语义的 Mask；经审核后接入。不能把SAM可见区域当作被挡住的完整形状。
2. 三维适配器：输入已验证单位、装配关系、内外参、接触和光源，输出真实投影/渲染及误差；复用同一六Mask/背景保护/验收。
3. 原生inpaint：实现capability与真实mask上传/极性/尺寸验证；保留本地保护像素门禁。
4. 透明/软体：增加透射背景、折射、厚度或重力形变求解及相应标注语义；当前二值不透明路径不代替这些计算。
5. 批处理：上层调度调用单任务流程，共享总预算与去重哈希。每样本独立manifest，禁止把一次模型高分当作全批有效。
6. 消融：`ablation-matrix JOB --output work/experiments.json` 只导出 E0–E7 计划；逐次改变一项并独立验收，结果尚未执行时保持 NOT_EXECUTED。

本次代码交付不包含新的付费图像生成，也没有声明两件真实物品已通过自然融合验收。
