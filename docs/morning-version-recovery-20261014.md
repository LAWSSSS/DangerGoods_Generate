# 10月12日上午版本溯源

两段源码由 Cursor 编辑记录中的完整文件内容快照恢复，来源不是 Git 分支的推测，也不是重新手写近似实现。仓库仍处于 `rollback/current-pose-scores` / `ec23f39`；测试通过独立加载恢复源码运行。

|版本|当日样本时段（北京时间）|源码完整 SHA256|恢复依据|
|---|---|---|---|
|前段：生图仅输入背景|09:54–10:11|c15b1f7495174e00b6c156a9f1575a1e5766b8a07cd74c6b0ee0d910f46bac29|09:26:39 编辑后的内容，与10:26:17首次加入参考图编辑前的内容完全相同|
|后段：生图输入背景和物品参考|10:32–11:38|f7e0260445880661ee7492cc6f6734f3c3f8e615f96a92888ca730b5b27fdf5e|10:27:35 最后一次成功编辑后的完整内容；随后历史运行命令启动批次|

两段各自的 `config.yaml`、负面提示词与共同的提示词模板也已恢复。完整文件编辑链在 [editor_provenance.json](/Users/liuzhaocheng/异物生成/output/validation/morning_both_three_objects_20261014_corrected/source_snapshot/editor_provenance.json)。

## 历史参数校对

21张上午照片元数据全部记录 `wan2.7-image` 与 `doubao-seed-2.1-pro`。历史命令指定 `--seed 20261012 --max-rounds 3 --total 1`；这是 Python 采样种子，原生 Wan 请求未传 seed。历史 CLI 命令在 [historical_cli_commands.json](/Users/liuzhaocheng/异物生成/output/validation/morning_both_three_objects_20261014_corrected/source_snapshot/historical_cli_commands.json)。

恢复的原生物品权重为：密封胶罐4、油缸3、减震器1。物品选择的随机数消耗顺序会影响后续尺寸和姿态，必须复用原生目录分类与加权选择。

使用旧元数据中的场景分析结果离线重放后，两段中的密封胶罐和油缸共4例，首轮尺寸、初始提示词、完整位置采样结果与朝向均逐项相同。密封胶罐首轮可见占宽0.03958660042307571，油缸0.40678414671336044。证据见 [manifest.json](/Users/liuzhaocheng/异物生成/output/validation/morning_both_three_objects_20261014_corrected/manifest.json)。重新请求场景分析与生图会有模型随机性，不能承诺输出相同图片。

## 本轮明确的调整

- 按用户选择测试密封胶罐、蓝色油缸、液压减震器；固定44、50；每组3轮；两版本共12组36张。液压减震器不在原上午批次中。
- 原代码只在尺寸偏大或偏小时继续，按用户要求只取消提前退出。AST比对确认循环其余结构相同：[loop_ast_crosscheck.json](/Users/liuzhaocheng/异物生成/output/validation/morning_both_three_objects_20261014_corrected/source_snapshot/loop_ast_crosscheck.json)。输出保留最后一轮，无下午版本的三项评分与最高平均分选择。
- 后段只提供指定主参考，继续遵守蓝色油缸不混入红色油缸的指令。原上午油缸曾同时输入蓝、红两张参考，因此本轮输入范围与该历史照片有差异。
- 两段都使用完整原背景，不输入上一轮生成图；不注入 Wan seed。增加日志与独立评估，独立评估不反馈到原生对话。
- 首次测试入口错误地以权重1选取物品，改变首轮尺寸；已终止，6张返回图全部剔除，保留在 [discarded_run.json](/Users/liuzhaocheng/异物生成/output/validation/morning_both_three_objects_20261014/discarded_run.json)。修正后新增历史确定性样本检查，正式批次重新执行。
