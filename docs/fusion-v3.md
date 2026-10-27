# V3 增量改造说明：程序 0.6.0 / Pipeline 3.0 / Prompt 3.0

日期：2026-10-16。依据用户提供的 V3.0 Markdown 与 ZIP，继续修改当前仓库。

## 1. 本版范围

本版落实可落地的诊断、图层合成、部件分割与修复路由。手册中的完整三维重建、专用重照明网络、自适应相机模型、所有类别/场景验证尚未实现；不能把参考包的设计目标或示例测试成绩当成本仓库的实测成绩。

| 项目 | 本版实际行为 | 证据边界 |
|---|---|---|
| 历史基线 | 保留所有旧输出；新增离线 `reprocess` | 固定候选和轮廓，另存结果；历史 Mask 缺少当时独立哈希，适配器仅允许 A 模式 |
| 独立实体层 | object/core/edge 分开，默认实体全部不透明 | core 是二值估计轮廓的形态分区，仍为 M2，不升级为审核真值 |
| 孔洞 | 分割保留负空间，额外保存封闭负空间诊断 | 封闭负空间不等于已确认机械孔洞 |
| 细部轮廓 | 可选部件独立颜色模型 + 原尺度局部 GrabCut | 不用计划代理直接填充，不通过形态闭运算补连接；仍会误分 |
| 接触层 | 独立二维接触核验后生成窄带线性照度衰减 | 已知支撑失败时不补黑影；二维接近不是承重证明 |
| 投射层 | 可信配准、可靠 H0、合法 receiver 内提取照度比 | 纹理/颜色不一致或交叉表面复杂时 UNKNOWN；不复制生成背景贴片 |
| 图层合成 | 原背景→衰减→物体→原前景→保护 RGB8 | 线性光空间计算；未授权区域精确还原 |
| 颜色处理 | 保留 H0 与已有按表面浮点 LAB 有界修正 | 不能移动错位高光，不能把材料颜色改成附近钢板均值 |
| Prompt | 九段通用合同 + 每件身份/材质/接触要求 | 只使用现有图像参考/bbox 能力，不声称原生 mask/depth 控制 |
| VLM 修复 | 缺陷分类、实例、ROI、证据、最多两项优先反馈 | UNKNOWN 转待复核；建议性重生不具备几何锁定 |
| 写盘门禁 | PNG 重解码校验原尺寸、protect、前景及实际授权层 | 像素契约通过不能证明视觉自然 |

源码在 `src/fusion_engine/controlled/fusion_v3/`，通过开关接入。旧任务未声明开关时继续使用原路线。

## 2. 三种版本各自负责什么

- 程序版本 `0.6.0`：本次实现。
- 输入任务 Schema `2.0`：保留原合同，增加可选字段；JSON Schema 镜像同步更新。
- `pipeline_version="3.0"` 与 `feature_flags`：显式选择新路线。
- `prompt_version="3.0"`：新的通用模板和逐物品编译。

ZIP 中 `example_v3_task_design.json` 的顶层结构是设计示例，不能直接传给现有 CLI。这里提供了实际可加载的 `config/controlled/v3/S01.json` 与 `S20.json`。

```json
{
  "schema_version": "2.0",
  "pipeline_version": "3.0",
  "prompt_version": "3.0",
  "feature_flags": {
    "v3_layer_compositor": true,
    "v3_mask_refinement": true,
    "v3_interaction_layers": true,
    "v3_diagnostics": true,
    "v3_repair_router": true
  }
}
```

上面只是现有任务的增量字段；完整任务仍需背景、物品、支点、Mask、预算等原有字段。设置不一致、未知开关或未实现的能力应报错，不静默忽略。

## 3. 算法与诊断顺序

1. 校验资产、同一坐标空间、完整投影、编辑/保护范围和阴影授权。
2. 生成或导入完整候选。原始候选字节与归一化另有记录；禁止裁切原图或拉伸不同长宽比。
3. 使用原有背景特征拟合小幅二维配准。V3 需要 `REGISTERED`，UNKNOWN 时停止依赖坐标的合成。
4. VLM 观察只给当前候选的形状、孔洞、部件与种子建议；观察用局部图不改变最终画幅。
5. 原有 GrabCut 给出 M2 可见轮廓；可选部件细分为不同小表面独立建颜色模型。部件前/背景点统一映射回完整原图，不执行图中文字。
6. 原前景、实体、孔洞、接触与投影不能混用一个 Mask。粗语义多边形不作为精确硬裁切边界；部件孔洞明显越出该部件时不强制挖除。
7. H0 仅校正公共背景漂移；保留原阈值和增益上限。H0 不可靠时，新路线跳过它并记录；依赖 H0 的投影提取同时 UNKNOWN。
8. 运行已有按材质分区的彩度/亮部有界调整与局部成像处理；不对全背景调色。
9. 单独建立接触与投影透过率，再在线性光空间将它们作用于原背景纹理。重复覆盖取更强的同类衰减，不把同一暗部叠黑两次。
10. 合成不透明实体，恢复原前景和 protect。写 PNG，重解码检查实际授权区外变化为 0。
11. 保留 VLM 意见、未知项与人工审核状态。缺件/错位高光不由调色假装修复。

### 阴影提取的限制

`fusion_v3.shadow_receivers` 中的 `receiver` 描述接收表面，`allowed` 描述被授权改变照度的区域，二者都不是已测出的真实阴影。只有二者交集内、实体/前景之外的可见表面可用。

程序先使用可靠 H0 拟合消除公共漂移，再分析线性亮度比，排除亮化、过黑/过曝、色相异常和对数高通纹理不一致。局部联合双边滤波只取同一有效连通域的样本，减少跨越结构边界的平滑。

当前显式保守参数：最大暗化 0.22，最小暗化 0.02，色度分量最大误差 0.035，对数纹理最大误差 0.12，最少 128 有效像素、有效比例 0.35、小连通域至少 16 像素。它们是开发参数，不是物理常数或跨场景已校准值。

没有可靠阴影时保持透过率为 1，状态 UNKNOWN；不能降低阈值只为了让示例出影子。复杂交叉面或未知接收面当前不自动求投影。

### 边缘与细结构

原版内缩羽化会把应当不透明的实体边缘与原背景混合。新内核默认二值不透明，不以分割置信度充当 alpha。内核可接收真实覆盖 alpha，但当前任务未接入专用 matting 或去污染模型；不声称解决全部锯齿/白边/黑边。

阀门由整体颜色模型分割时，较大的阀体可能压过面积很小的手轮和细杆。新路线给每个观察部件独立建立颜色模型，使用当前候选支持的颜色和可选部件点作种子，保留语义边界外的有限分析上下文。颜色相近、过曝、孔洞模糊等仍可能导致 UNKNOWN。轮廓的像素连通也不能证明实际机械连接正确。

## 4. 使用方式

在项目根目录使用现有 Python 环境。模型与密钥仍从私有 `config_bailian.yaml` 读取，未写入任务 JSON。38 件资产目录与原图均不修改。

### 离线校验、规划与提示词

```bash
python -B fusion.py validate config/controlled/v3/S01.json
python -B fusion.py plan config/controlled/v3/S20.json --output work/my-v3-plan
python -B fusion.py prompt config/controlled/v3/S01.json --output work/my-v3-prompt
python -B fusion.py request-preview config/controlled/v3/S01.json --provider-config config_bailian.yaml --output work/my-v3-request.json
```

以上不发模型请求。A 的 READY 只表示可生成概念预览，计划中的承载力/尺度/前景分割仍可能 UNKNOWN。不是已验收。

### 固定历史候选与 Mask 的合成对照

```bash
python -B fusion.py reprocess config/controlled/v3/S20.json \
  --source-run output/controlled-v2_3-final-check/final_reprocess_v2_3_S20/20261015T101405Z-8f30ab3a \
  --ablations
```

这条路线不导入提供商配置，没有网络调用。即使当前配置启用了新分割开关，默认仍冻结历史 Mask，`source.mask_policy` 会明确记录此行为。

### 单独比较部件细分

```bash
python -B fusion.py reprocess config/controlled/v3/S20.json \
  --source-run output/controlled-v2_3-final-check/final_reprocess_v2_3_S20/20261015T101405Z-8f30ab3a \
  --ablations --refine-masks
```

复用候选绑定的历史观察，重新估计轮廓，仍为零模型请求。输出额外的 `A4_v3_with_frozen_old_mask`，与同次新轮廓结果比较；不能把“合成+分割一起变化”说成纯合成收益。

### 使用配置中的 VLM 观察导入候选

```bash
python -B fusion.py run config/controlled/v3/S20.json \
  --candidate <完整候选PNG路径> --provider-config config_bailian.yaml
```

只调用预算内的观察/辅助审阅，不调用生图。若明确需要生成，使用 `--generate` 替代 `--candidate`。当前 Wan 路线只有参考图和 bbox 提示，没有原生二值 inpaint、depth/normal 控制或几何锁定。修复建议只有在缺陷明确且预算足够时才继续指导重生；未知优先转复核。

## 5. 输出如何阅读

每次输出另建目录，已有结果不覆盖：

```text
manifest.json
job.snapshot.json
analysis/
  placement.json
  candidate-001.aligned.png          # 在线导入/生成路线
  candidate-001.observation.json
  candidate-001.geometry.json
  candidate-001.layers/
    V0_original.png
    V2_aligned_candidate.png
    V3_adjusted_object.png
    V4_object_only.png
    V5_final_layers.png
    mask_object.png / mask_core.png / mask_edge.png
    mask_front.png / mask_contact.png / mask_cast.png
    mask_edit.png / mask_protect.png
    mask_enclosed_negative_space.png
    cast_transmission.npy
    contact_transmission.npy
    interaction_attenuation.png
    difference_heatmap.png
    analysis_only_200pct_*.png
    diagnostics.json
  candidate-001.layers.authorized.png
outputs/candidate-001.png
```

离线 `reprocess` 的图层目录名为 `analysis/layers/`。200% 放大和热图仅供诊断；`outputs/` 的母版保持完整原尺寸。浮点透过率单独保存，白/1 表示不暗化；这不是物理真实透光率。

`--ablations` 另外保存历史 v2、无外观处理、无新增 AO、局部候选贴片诊断、V3 最终图。贴片组专门用于观察丢失交互的原因，不可直接接受为正式样本。各对照的种子/候选/M2/交互场应按 `source` 和运行说明解释。

## 6. 回滚

使用原 `config/controlled/v2_3/S01.json` 或 `S20.json`。旧模板、旧配置和历史输出保留。新配置中的 V3 从属开关不能在关闭主合成器后仍被静默使用；如果手动迁移回旧路线，应同时移除 V3 增量字段。

开始本次改动前另存了本地源码/配置基线到 `work/fusion-v3-baseline-20261016/`；不含私有配置和密钥。未进行提交、推送或远程发布。

## 7. 验收边界与后续工作

已确认的像素契约、人工视觉判断、VLM 意见与训练标注必须分开记录。`CONCEPT_ONLY` 和零保护区变化不能写成“已真实融合”。

下一阶段需要针对原候选的错位高光、细部孔洞和真实承托继续提供可靠证据：可审核的前景/孔洞分割、近同材质且同朝向的照明参考，或实际支持局部受控编辑的模型。当前不把这些缺失能力伪装成 prompt 已经执行的内部步骤。

参考文件指纹见 `docs/specifications/v3-reference-manifest.json`。外部 Markdown 与 ZIP 内同名 Markdown 字节一致。参考包声称的 10 项单元测试为作者记录，本次未把它作为本仓库测试通过的证据。

## 8. 本轮真实处理记录

本轮保存10个独立运行目录，其中2个调用了配置中的VLM，各2次；其余为离线对照。**总计0次生图、4次VLM请求**。

| 最终离线预览 | 完整尺寸 | 保护区变化 | 实际授权图层外变化 | 自动投影 | 状态 |
|---|---|---|---|---|---|
| S01 | 3840×2160 | 0像素 | 0像素 | UNKNOWN | CONCEPT_ONLY，待视觉复核 |
| S20 | 3840×2160 | 0像素 | 0像素 | UNKNOWN | CONCEPT_ONLY，待视觉复核 |

最终两张图均绑定当前源码指纹。旧2.3路线另按相同候选/M2重处理，与对应历史结果逐像素差异均为0。阀门同一输入的后续重处理母版SHA-256与先前成功处理一致。上述为具体样本的确定性与兼容证据，不代表不同物品/场景已普遍验证。

阀门新观察有效，部件点帮助恢复一些被删掉的轮缘和细连接；最终辅助审阅失败而保留UNKNOWN。气瓶新观察未返回有效JSON，该轮被拒收；最终离线母版复用历史有效候选绑定观察。气瓶那次成功审阅的对象是未后处理候选，不能用于宣称最终离线图合格。

Ruff、43个包内Python源码解析、Schema镜像及差异空白检查已执行。新Prompt没有重新生图A/B，投影提取在这两个候选上均没有足够证据，未声称完成物理重照明、所有V3阶段或自然度验收。

本轮交付、逐运行记录与两张原尺寸图位于 `output/review-v3-20261016/`。失败记录与中间对照均保留。
