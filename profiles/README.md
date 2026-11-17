# 逐件尺寸与 Prompt 配置

版本：**1.0.0**；日期：2026-10-14。

已覆盖桌面六视图中的 38 件物品（S01–S38）和背景目录中的 73 张图片。当前项目生成入口为 `generate_aliyun.py`，通过 `config.yaml` 的 `use_object_catalog: true` 启用本目录。

## 文件索引

| 文件 | 用途 |
|---|---|
| `物品尺寸与Prompt逐件手册.md` | 38 件总表、尺寸依据、逐件完整 prompt 与使用说明 |
| `object_profile_seeds.json` | 编辑入口：主体/零件尺寸、范围、材质、姿态、遮挡、身份特征 |
| `object_profiles.json` | 运行参数，包括尺寸证据、比例范围及完整 prompt |
| `object_prompts.json` | 按物品 ID 查询配套 prompt |
| `prompts/S01.txt` … `S38.txt` | 每件可直接复制的独立提示词 |
| `background_profiles.json` | 背景分组、分辨率、哈希、亮度统计与保护区域提示 |
| `validation_report.json` | 本次离线检查记录 |

## 修改与使用

在项目根目录运行：

```bash
python object_catalog.py --list
python object_catalog.py --item S15
python object_catalog.py --item S26 --output output/item_prompts/S26.txt
```

修改 `object_profile_seeds.json` 中指定 ID 的工作值后，在项目根目录运行：

```bash
python tools/build_object_profiles.py
```

这会同步重建运行参数、占比范围、38 份文本及逐件手册。直接编辑派生文件的改动会被重建覆盖。图片目录改变时，同时更新种子的 `reference_root`、`background_root` 和项目 `config.yaml` 的 `six_views_dir`。

真实生成时可指定一件：

```bash
python generate_aliyun.py --objects S15 --total 1
```

多件 ID 的现有入口采用加权随机选择；本次配置任务未启动批量生成。API 凭据沿用项目现有 `config_bailian.yaml`，本目录与下载包均不包含凭据。

## 尺寸与效果的证据

大部分尺寸是用于合成的工作估计值，尚未测量具体实物。S35 使用 UCP210 型号目录的条件尺寸，仍不属于实例测量。背景水平跨度 2.4m 沿用旧程序假设，未做现场标定。

画面比例同时考虑投影和遮挡；大型储罐通过合理遮挡或局部出画保持实体尺度。背景 ROI 是待逐图确认的区域建议，不是已经制作的 mask。背景保护、接触、阴影与材质一致性目前通过 prompt 约束，不能据此宣称像素级保持或生成效果已验收。

本次完成了素材清点、逐件参数编写、程序接入和离线检查，没有调用生图/VLM 模型。离线检查验证配置与代码行为，不能替代真实生成后的视觉验收。

## 1.0.0 版本说明

- 为全部 38 张六视图建立唯一物品 ID、独立尺寸与完整 prompt。
- 将 DN25/DN100/DN200/DN350 阀门、大小压力罐、不同千斤顶和减震器分开配置。
- 分别设置钢、铸铁、不锈钢、铝、塑料与橡胶部件的旧化方式。
- 只给模型当前选中物品的六视图，不混入同类其他型号。
- 将尺寸证据和背景未标定状态加入参数及 VLM 评估提示词。
- 保留原始素材、既有 API 配置和旧类别入口；新的物品配置位于 `profiles/`。
