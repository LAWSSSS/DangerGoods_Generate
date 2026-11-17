# 原尺寸范围分支：全类别三轮迭代对照

日期：2026-10-14。测试工作分支：`rollback/prev-size-range`，提交 `b65cc9cc4ee5e57c05a1d2bd612298951ee2a67c`。

## 已确认范围

- 用户指定背景44、50；每类一张主参考。
- 15类×2张固定背景×3轮，30组，Wan最多90次请求，每次1张。
- 生图 `wan2.7-image`；原方案评估和独立对照均为 `qwen3.8-flash`。
- 两种减震器原照片在原分支均归为“减震器”，此次选 `液压减震器.jpg`；其他主参考在实验 `manifest.json` 逐项登记。

## 回退与保存

此前 `rollback/current-pose-scores` 的未提交源码、配置、文档、检查和试验记录保存在Git stash，未提交到分支，也未推送。完整变化补丁、未跟踪文件哈希、两个提示词文件原始字节及固定stash标识另存于 `output/rollback_preservation/20261014102017/`。

提示词目录原为只读，切换时临时允许目录写入，检出完毕恢复原权限。旧分支的 `generate_aliyun.py`、`config.yaml`、提示词与依赖文件已逐字节核对，与提交完全一致。本轮只新增独立测试脚本、离线脚本检查和此报告。

## 测试方式与边界

旧链路首轮仅输入固定原背景；第二、三轮输入固定原背景和上一轮生成图。**原物参考照片不进入旧方案的生图或原尺寸评估。** 它用于本轮新增的独立结构/物理对照，不会反馈到旧方案，避免改变被测试的算法。

旧方案在尺寸合格或不要求重试时会提前退出。本轮测试脚本仅抑制这一退出条件，使每组保留完整三轮；原评估结果、原本会在哪轮退出的条件、每轮反馈和缩放过程均保存。没有把合格评价改成失败来强制迭代。

测试传输层避免输出密钥，并关闭POST自动重试。每次Wan提交前记账，同一已提交轮次不得重发；中断后不会把未知结果作为未请求重新提交。固定基准种子20261014，各类种子为基准加类别索引，两个背景及该类三个生成轮次使用同一个Wan种子；同种子不保证模型输出绝对一致。

每轮另做一次独立Qwen对照，输入原物照片、原背景、完整生成图和计划位置的原像素局部，检查：新增目标、主体/颜色/口部/连接、落点、支撑、前景遮挡、尺度及成像自然度。目标偏移时完整图仍是判断依据，不能仅凭局部没有物品就判缺失。

只有各项明确通过、证据完整、没有不确定项且置信度至少0.65，才记为独立通过。原尺寸评估可能将缺失或API失败归为“合格”；实验会单独标记评估是否有效，默认回退不能视为通过。独立模型判断仍不是三维测量或人工验收。

## 运行与结果入口

完整实验目录：`output/validation/prev_branch_all_objects_20261014/`。

- [动态逐轮对照画廊](../output/validation/prev_branch_all_objects_20261014/index.html)
- [全类别状态与三轮尺寸判断](../output/validation/prev_branch_all_objects_20261014/summary.md)
- [完整汇总及独立评估](../output/validation/prev_branch_all_objects_20261014/report.json)
- [主参考、固定场景、种子及源码哈希](../output/validation/prev_branch_all_objects_20261014/manifest.json)
- [实际请求账本](../output/validation/prev_branch_all_objects_20261014/requests.json)
- [测试脚本的离线检查](../output/validation/prev_branch_all_objects_20261014/harness_checks.log)

每组 `cases/<id>/` 保存三轮完整API图片、目标局部、逐轮实际提示词、参数、输入来源、API完整响应、两种评估、模型完整响应及原反馈历史。API图片按原返回字节保存，展示裁图另存。源码快照不包含密钥配置。

```bash
# 实际运行：只处理尚未启动的组
.venv/bin/python scripts/test_prev_branch_iterations.py \
  --run-dir output/validation/prev_branch_all_objects_20261014 --workers 2

# 不请求模型，刷新报告和画廊
.venv/bin/python scripts/test_prev_branch_iterations.py --report-only \
  --run-dir output/validation/prev_branch_all_objects_20261014
```

此文先保存试验范围与方法；最终数量、逐轮效果和异常以请求账本及汇总报告为依据。未完成或失败的组不记作完成，也不把“完成三轮”写成质量验收通过。
