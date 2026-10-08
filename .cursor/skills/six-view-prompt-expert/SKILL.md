---
name: six-view-prompt-expert
description: Image-processing expert that inspects one object photo with doubao-seed-2.1-pro and writes super-detailed front/back/left/right/top/bottom 6-view prompts, each view separately. Use when the user imports an image, asks for 六视图 prompt, 前后左右上下, orthographic turnaround, or 6-view generation prompts.
---

# 六视图 Prompt 专家

定位：图片处理专家。不靠猜测写方位，必须先把原图交给 `doubao-seed-2.1-pro` 看图。

## 何时用

用户导入一张异物/产品图，或要求生成前后左右上下六视图 prompt 时，先跑本流程，**确认 prompt 后再生图**。

## 工作流

1. 确认图片路径（默认可从 `input/objects/` 取）。
2. 执行：

```bash
python describe_six_views.py --image <图片路径>
```

3. 把 `output/six_views/<stem>/六视图prompt.md` 全文给用户看。六个视角必须分开，每一格都写清突出件在**本画面**的方向。
4. 用户确认后，才调用现有生图脚本（如 `generate_six_views.py --stem <stem> --force`）。未确认不生图。

## 硬规则

- VLM 固定 `doubao-seed-2.1-pro`，密钥和地址只从 `config_bailian.yaml` 读，不要改成 qwen。
- 参考图经常是 3/4 透视：照片里「朝右」不等于物体朝右。必须先建刚体坐标系，再把每个零件投影到六个正交相机。
- 每个视角单独写：相机、可见面、每个突出件（握把/喷口/软管/阀门/手柄等）的画面方向、被遮挡件、禁止画成哪一格。
- 禁止六个格子共用一句含糊描述。禁止只写「保持一致」。

## 输出

脚本会写：

- `output/six_views/<stem>/六视图prompt.md` 给人看
- `output/six_views/<stem>/六视图prompt.json` 给生图程序用
