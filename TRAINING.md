# 训练

在训练页准备数据并启动训练

## 数据准备

支持 WAV、FLAC、OGG、MP3 音频目录中每段录音配同名txt：

```text
voice_data/
  001.wav
  001.txt
  002.wav
  002.txt
  hotaru.wav
  hotaru.txt
```

文本要和录音内容对应 也可以在“标注 CSV”填 CSV 路径，`file_name` 相对音频目录解析，也支持绝对路径

```csv
file_name,text,caption,speaker_id
001.wav,こんにちは。,穏やかに話す。,voice_a
002.wav,また会いましたね。,少し嬉しそうに話す。,voice_a
```

`file_name` 和 `text` 必填，`caption`、`speaker_id` 留空时使用页面上的默认演技和说话人 同一个人的录音使用相同说话人标识，VoiceDesign 数据最好还是标一下演技

1. 填音频目录及可选的 CSV，点击检查数据
2. 确认文本、时长和说话人，点击预处理。超过所设最长时长的录音会报错，需要把音频与对应文字一起切分，程序不直接截断录音
3. 预处理完成后，训练清单会自动填入 `manifest.jsonl`。已有的 Irodori latent manifest 也可直接填写

## 训练设置

| 方式 | 输出 | 继续训练 |
| --- | --- | --- |
| 音色训练 | `.speaker.safetensors` | 在`继续训练音色`填已有音色文件，Token 数需匹配，优化器和步数重新开始 |
| LoRA 微调 | 含适配器与 `trainer_state.pt` 的检查点目录 | 在`继续训练检查点`填该目录，恢复训练状态 |

默认batch大小为 1、梯度累积为 4，启用梯度checkpoint。音色训练默认学习率为 `0.01`，切换到 LoRA 时为 `0.0001`，开始前可以调整。训练精度控制前向计算，上游训练器还是FP32的权重和优化器状态

默认目录：

```text
tts_workspace/training/datasets/    latent 与训练清单
tts_workspace/training/runs/        yml、日志、ckpt
```
