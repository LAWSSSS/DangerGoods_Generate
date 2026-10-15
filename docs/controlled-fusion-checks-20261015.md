# 受控融合 0.3.0 本地检查记录

日期：2026-10-15。分支：`codex/manual-driven-scene-fusion`。
修改前本地快照：`26a705ee5115508061285cfab9673ce9b6eb21a8`。
本记录描述代码检查与离线操作，不表示真实物品的自然融合效果已验收。

## 1. 已执行检查

| 项目 | 实际结果 |
|---|---|
| 新增Python模块及入口语法/显式字节码编译 | 27个文件PASS；字节码写入work/manual-v2/bytecode，原只读目录未使用 |
| Ruff静态检查 | All checks passed |
| Ruff格式检查 | 27 files already formatted |
| JSON Schema定义 | 两份Draft 2020-12定义合法；Python定义与发布JSON资源一致 |
| 任务配置 | S01、S20、联合三份草案Schema均合法 |
| 人工审核格式示例 | Schema合法；零哈希/UNKNOWN/未确认，不能直接用于接受 |
| 原始手册副本 | 与Downloads原始文件字节相同 |
| 本地提供者读取 | 模型wan2.7-image、VLM doubao-seed-2.1-pro；只显示脱敏公开配置 |
| 实际请求参数离线构造 | 2张生锈参考+1张原背景；区域框属于原生bbox_list；n=1、2K、水印false；未发送请求 |
| 尺度区间工具 | 示例参数得到134.4–277.2px；缺少深度/方向比时UNKNOWN |
| 原手册meters参数兼容 | 示例得到218.1818px的显式退化区间；不作真实标定声明 |
| E0–E7 | 成功导出8项NOT_EXECUTED消融计划；没有执行真实生图消融 |

编译与离线检查使用现有本机Python环境，没有安装/替换用户环境。

## 2. 当前真实资产的规划结果

| 草案 | 状态 | 当前缺失/冲突项数 | API调用 |
|---|---|---:|---:|
| rusted_S01.draft.json | INPUT_REQUIRED | 12 | 0 |
| rusted_S20.draft.json | INPUT_REQUIRED | 12 | 0 |
| rusted_S01_S20.draft.json | INPUT_REQUIRED | 23 | 0 |

主要缺少完整轮廓、原前景、接触交互区、确认后的完整投影和有锚定的承载/稳定性证据。
增加接触区域与支撑链检查后，联合草案的问题项从早期19项增为23项；不是已完成图像的质量分数。
目录迁移保留尺寸estimated状态，没有继承2.4m料斗跨度作为已标定事实。

在联合草案上执行候选导入入口后，运行停在INPUT_REQUIRED，image_calls=0、vlm_calls=0、candidates=0。
产物目录：`output/controlled_fusion/catalog_S01_S20/20261015T050637Z-825b7c50`。

## 3. 离线回放诊断的含义

为了检查运行目录、预算账本、PNG保存、质量报告和审核模板的连接，使用原背景本身作为未修改候选进行一次A模式回放。
该操作没有插入物体，不能证明融合真实性、旧化效果、支撑几何或图像模型调用成功。
结果为CONCEPT_ONLY，image_calls=0、vlm_calls=0、candidates=1，没有outputs/final.png或已验收标注。

目录：`work/manual-v2/offline-diagnostic/offline_import_diagnostic/20261015T050637Z-4ddc3db1`。

对这张回放PNG运行原手册兼容参数的像素工具：

- 全保护范围8,294,400像素：max_channel_delta=0、changed_pixel_count=0、PASS。
- 空保护范围：applicable=false、UNKNOWN、exact_pixel_lock_claim=false、退出码4。

上述零差分仅验证未修改回放的RGB8/PNG链路；没有把同图比较当作真实融合案例合格的依据。

原背景实际为3840×2160 JPEG，无EXIF旋转和ICC。请求预览记录了模型输入编码后的尺寸与坐标映射，生产输出配准策略必须显式选择。

## 4. 审计与版本边界

实际运行写入输入哈希、程序源码指纹、依赖版本、实际请求摘要、预算和阶段事件。
人工审核绑定最终PNG哈希并冻结实际轮廓；实际标注不从计划Mask自动冒充生成结果提取。
此次检查没有新的图像/VLM网络调用。原生二值inpaint、自动可靠amodal分割、三维力学/重建、玻璃折射和软体变形仍需接入真实专用能力。
先前两张真实生锈物品测试没有因本次代码检查被重新标记合格。
