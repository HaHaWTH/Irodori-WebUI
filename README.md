# Irodori TTS WebUI

Irodori-TTS v3 的中文 WebUI

## 安装

1. 安装 `requirements-ui.txt` 内的依赖 使用 python 3.10
2. 按 [Irodori-TTS v3](https://github.com/Aratako/Irodori-TTS/tree/v3) 的说明准备推理环境
3. 把v3的仓库克隆下来 `git clone --branch v3 --depth 1 https://github.com/Aratako/Irodori-TTS.git`
4. 丢进`.runtime/`，或用 `IRODORI_RUNTIME_DIR` 指向已有目录
5. 下载模型 `python download_models.py`
6. `python app/webui_tts.py --server-port 7862`

`--no-browser` 关闭自动打开浏览器

## 音色

- **音色文件**：在音色管理上传
- **参考音频**：上传单人干声直接推理
- **音色设计**：根据演技描述生成
- **LoRA**：在模型设置指定兼容的适配器目录

## 批量生成

| 字段 | 行为                                                |
|------|-----------------------------------------------------|
| 名称 | 可选 用于输出文件名                                 |
| 台词 | 必填                                                |
| 演技 | 留空继承上句 `[none]` 清空演技 之后继承清空后的状态 |
| 种子 | 留空使用全局种子 全局也为空时随机生成               |

### 示例
```csv
Name,Text,Style,Seed
First,こんにちは。,優しく穏やかに話す。,114514
Second,また会いましたね。,,
```

## 目录

```text
tts_workspace/voices/     导入的音色
tts_workspace/outputs/    每批 WAV JSON ZIP
tts_workspace/tables/     导出的台词表
```