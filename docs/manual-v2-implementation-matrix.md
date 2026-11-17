# 手册 V2.0 与实现状态映射

实现版本0.3.0，2026-10-15。状态定义：**已实现**为可执行代码；**需外部证据**为可接收/核查但不自动捏造输入；**待专用适配**为尚未实现的物理/模型能力。

| 手册要求 | 代码 | 状态与实际边界 |
|---|---|---|
| S00 需求/模式 A/B/C | controlled/schema.py、job.py | 已实现；B/C强制保护，不静默降级 |
| S01 原图登记/哈希 | job.py、workflow.py | 已实现；运行前后/验收时校验来源 |
| S02 结构/多视图/真实尺寸 | catalog.py、schema.py | 已实现目录/部件结构；多视图重建需专用适配 |
| S03 场景区域/尺度/光照 | planning.py、appearance.py | 区域配置与统计已实现；语义/深度/主光自动估计未执行 |
| S04 三类尺度来源/不确定性 | geometry.py | 四等级已实现；未知比值不补1；标定需真实输入 |
| S05 OBB/针孔投影 | geometry.py、planning.py | 已实现；三维点由外部提供；未自动拟合CAD |
| S06 支撑/接触/重心 | planning.py、quality.py | 2D间隙/凸包已实现；承重/摩擦/3D稳定性需证据 |
| S07 前后关系/局部循环 | masks.py、geometry.py | 已实现逐区域/像素DAG；未覆盖交叠UNKNOWN |
| S08 六Mask与极性 | masks.py | 已实现；显式轮廓/多边形；缺失不生成伪标签 |
| 背景/叠字/原前景保护 | masks.py、appearance.py、quality.py | 已实现最终恢复与解码PNG零差分；空范围UNKNOWN |
| S09 多基材/涂层/旧化 | appearance.py、planning.py | 16种材质的分部件计划/Prompt已实现；不是PBR模拟 |
| S10 光照/局部统计/PSF | appearance.py | 有界LAB、局部模糊/噪声已实现；光照方向用证据/审阅 |
| S11 各种生图模型 | controlled/adapters.py、providers/ | 实际Wan/Seedream协议；其他输出可导入；新协议需实现适配器 |
| S12 局部修复/原生inpaint | workflow.py、adapters.py | 预算内最小修复提示已实现；当前provider无二值mask/depth控制 |
| S13 禁止全图滤镜掩错 | appearance.py | 生产路径只写目标/编辑区；历史参数仅独立工具 |
| S14 最后恢复原保护像素 | appearance.py | 已实现；质量基于保存并重新解码后的母版 |
| S15 硬约束/未知/独立几何 | quality.py | 已实现；不平均抵消硬失败，不以VLM唯一放行 |
| S15 哈希绑定人工审核 | workflow.py、job.py | 已实现；实际轮廓冻结，不覆盖历史审核 |
| S16 PNG/实际标注/可追溯 | workflow.py、quality.py | 已实现；COCO兼容RLE和amodal扩展仅来自已审核实际轮廓 |
| 逐物品/逐场景作业配方 | 旧profiles与新catalog.py | 38件可逐个迁移，当前三份S01/S20草案；任意背景显式输入 |
| 玻璃折射、软体重力 | planning.py | 待专用适配；当前状态SPECIAL_RENDERER_REQUIRED |
| 历史两轮+LAB参数 | 原手册、utilities.py | 保留历史说明；未重跑历史生成或伪称当前哈希复现 |
| E0–E7消融/验收记录 | catalog.py、cli.py | 实验计划生成已实现；真实消融尚未执行 |
| JSON Schema/配置/伪代码 | schemas/、config/controlled/、workflow.py | 可执行实现已交付；配置草案不构成输入已确认 |

## S01 / S20 当前样本状态

两件物品使用用户生锈六视图，物理尺寸保留先前工作值的 estimated 状态。
联合和单件草案仍需原图中的完整投影、前景遮挡、接触区、承载证据及确认后的摆放。
当前离线规划返回 INPUT_REQUIRED，模型调用为0。此前两图测试失败不因本次重构被重新标为合格。

## 验收不代表的能力

本程序的 ACCEPTED 代表当前契约中的像素、几何及人工检查通过。
它不证明图像是真实相机拍摄，不证明输入尺寸已经实测，也不证明获得了生成模型隐藏的内部状态。
用于训练/评估时应保留 synthetic 标记、来源划分与独立真实样本验证计划。
