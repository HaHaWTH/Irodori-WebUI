#!/usr/bin/env python3
from __future__ import annotations

from tts_bootstrap import CACHE, DATA, ROOT, RUNTIME

import argparse
import threading
from pathlib import Path

import gradio as gr

from tts_engine import (
    DEFAULT_CODEC, DEFAULT_MODEL, DISPLAY_HEADERS, Settings, Workbench, clean_path,
    export_rows, import_rows, import_voice, normalize_rows, split_text, voice_choices,
)
from tts_theme import CSS, make_theme
from tts_paths import path_field
from tts_training import GPU_LOCK
from training_ui import build_training_tab
from irodori_tts.inference_runtime import (
    default_runtime_device, list_available_runtime_devices, list_available_runtime_precisions,
)

VOICES = DATA / "voices"
ENGINE = Workbench(DATA / "outputs")
ACTIVE: dict[str | None, threading.Event] = {}
SOURCE_CHOICES = [("音色文件", "Speaker file"), ("参考音频", "Reference audio"), ("音色设计", "Voice design")]
STATUS_LABELS = {"Pending": "等待", "Generating": "生成中", "Done": "完成", "Failed": "失败",
                 "Stopped": "已停止", "Skipped": "已跳过"}
HEADER = """<div class="app-heading"><svg class="app-mark" viewBox="0 0 32 32" aria-hidden="true">
<rect width="32" height="32" rx="9" fill="currentColor" opacity=".12"/>
<path d="M8 14v4m5-10v16m6-13v10m5-7v4" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
</svg><h1>Irodori TTS</h1></div>"""


def library_rows():
    return [[label, Path(value).name, round(Path(value).stat().st_size / 1024, 1)]
            for label, value in voice_choices(VOICES)]


def refresh_voices(current):
    choices = voice_choices(VOICES)
    values = [value for _, value in choices]
    selected = current if current in values else (values[0] if values else None)
    return gr.update(choices=choices, value=selected), library_rows()


def add_voice_file(upload):
    if not upload:
        raise gr.Error("Select one or more speaker files.")
    imported, failed, errors = [], [], []
    for source in upload:
        try:
            imported.append(import_voice(clean_path(source), VOICES))
        except Exception as exc:
            failed.append(source)
            errors.append(f"{Path(source).name}: {exc}")
    if not imported:
        raise gr.Error("No speaker files were imported.\n" + "\n".join(errors))
    if failed:
        gr.Warning(f"Imported {len(imported)} speaker file(s); {len(failed)} failed.\n" + "\n".join(errors))
    else:
        gr.Info(f"Imported {len(imported)} speaker file(s).")
    return gr.update(choices=voice_choices(VOICES), value=imported[0]), library_rows(), failed or None


def add_voice_path(path):
    if not path or not clean_path(path):
        raise gr.Error("Select a speaker file or enter its path.")
    selection, rows, _ = add_voice_file([clean_path(path)])
    return selection, rows, ""


def enable_if_present(value):
    return gr.update(interactive=bool(value and str(value).strip()))


def change_source(source):
    return (gr.update(visible=source == "Speaker file"), gr.update(visible=source == "Reference audio"),
            gr.update(visible=source != "Voice design"))


def change_device(device, model_precision, codec_precision):
    choices = list_available_runtime_precisions(device)
    default = "bf16" if "bf16" in choices else "fp32"
    return tuple(gr.update(choices=choices, value=value if value in choices else default)
                 for value in (model_precision, codec_precision))


def append_text(text, mode, rows):
    try:
        return normalize_rows(rows) + split_text(text, mode)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def append_and_clear(text, mode, rows):
    return append_text(text, mode, rows), ""


def append_file(upload, mode, rows):
    if not upload:
        raise gr.Error("Select a queue file.")
    try:
        return normalize_rows(rows) + import_rows(upload, mode)
    except (ValueError, KeyError, OSError) as exc:
        raise gr.Error(f"Queue import failed: {exc}") from exc


def download_rows(rows):
    try:
        return export_rows(rows, DATA / "tables")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def pack_settings(checkpoint, codec, device, model_precision, codec_precision, source, voice, reference, lora,
                  caption, steps, duration, speaker_cfg, caption_cfg, text_cfg, seed, trim_tail):
    return Settings(checkpoint=checkpoint, codec=codec, device=device,
                    model_precision=model_precision, codec_precision=codec_precision,
                    source=source, embedding=voice or "", reference=reference or "",
                    lora=clean_path(lora), caption=caption, steps=int(steps),
                    duration_scale=float(duration), speaker_cfg=float(speaker_cfg),
                    caption_cfg=float(caption_cfg), text_cfg=float(text_cfg), seed=seed, trim_tail=trim_tail)


def status_html(phase="ready", records=()):
    completed = sum(row["status"] == "Done" for row in records)
    failed = sum(row["status"] == "Failed" for row in records)
    processed = completed + failed
    total = len(records)
    labels = {"ready": "等待生成", "loading": "加载模型", "running": "正在生成",
              "finished": "生成完成", "stopped": "已停止", "error": "模型加载失败"}
    label = labels[phase]
    if phase == "finished" and failed:
        label = f"完成 {completed} · 失败 {failed}"
    progress = round(processed / total * 100) if total else 0
    count = f"{processed} / {total}" if total else ""
    return (f'<div class="run-status" data-phase="{phase}"><div class="run-status-line">'
            f'<span class="status-dot"></span><span>{label}</span><strong>{count}</strong></div>'
            f'<div class="run-track"><span style="width:{progress}%"></span></div></div>')


def render_snapshot(state, previous_clip=""):
    records = state["records"]
    table = [[row["index"], row["title"] or row["text"][:35], STATUS_LABELS[row["status"]],
              row["used_seed"], row["seconds"], row["error"]] for row in records]
    choices = [(Path(path).name, path) for path in state["files"]]
    latest = state["files"][-1] if state["files"] else None
    # Do not reload the same audio on every progress update
    changed = latest != previous_clip
    preview = gr.update(choices=choices, value=latest) if changed else gr.skip()
    audio = latest if changed else gr.skip()
    download = gr.update(value=latest, interactive=bool(latest)) if changed else gr.skip()
    archive = gr.update(value=state["archive"], interactive=bool(state["archive"]))
    return table, status_html(state["phase"], records), archive, preview, audio, state["log"], download


def generate(request: gr.Request, rows, continue_on_error, *values):
    if not GPU_LOCK.acquire(blocking=False):
        raise gr.Error("Training or preprocessing is running. Stop it before synthesis.")
    stop = threading.Event()
    ACTIVE[request.session_hash] = stop
    previous_clip = ""
    try:
        settings = pack_settings(*values)
        for state in ENGINE.run(rows, settings, stop, continue_on_error):
            yield render_snapshot(state, previous_clip)
            previous_clip = state["files"][-1] if state["files"] else None
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        ACTIVE.pop(request.session_hash, None)
        GPU_LOCK.release()


def generate_single(request: gr.Request, text, *values):
    yield from generate(request, [["", text, "", ""]], False, *values)


def select_clip(path):
    return path, gr.update(value=path, interactive=bool(path))


def stop_batch(request: gr.Request):
    stop = ACTIVE.get(request.session_hash)
    if stop is None:
        gr.Info("No active job.")
    else:
        stop.set()
        gr.Info("Stopping after the current item.")


def release_model():
    if not GPU_LOCK.acquire(blocking=False):
        raise gr.Error("The GPU is busy. Stop the current training task first.")
    try:
        ENGINE.release()
        gr.Info("Model unloaded.")
    finally:
        GPU_LOCK.release()


def build_ui():
    device_default = default_runtime_device()
    precisions = list_available_runtime_precisions(device_default)
    voices = voice_choices(VOICES)
    with gr.Blocks(title="Irodori TTS") as demo:
        gr.HTML(HEADER, elem_id="app-header")
        with gr.Tabs(elem_id="workspace-tabs"):
            with gr.Tab("语音合成", id="synthesis"):
                with gr.Row(elem_id="voice-strip"):
                    source = gr.Radio(SOURCE_CHOICES, value="Speaker file", label="音色来源", scale=1, min_width=340)
                    with gr.Column(scale=2, min_width=340, elem_id="voice-controls") as voice_controls:
                        with gr.Group() as voice_group:
                            with gr.Row(elem_id="voice-select-row"):
                                voice = gr.Dropdown(voices, value=voices[0][1] if voices else None,
                                                    label="音色", scale=5, min_width=200)
                                refresh = gr.Button("刷新", scale=0, min_width=72, elem_id="voice-refresh")
                        with gr.Group(visible=False) as reference_group:
                            reference = gr.Audio(label="参考音频", type="filepath", sources=["upload"],
                                                 editable=False, buttons=[])
                            reference_browse = gr.UploadButton("选择音频", file_types=["audio"], type="filepath")
                with gr.Row():
                    with gr.Column(scale=2, min_width=580, elem_id="editor-column"):
                        caption = gr.Textbox(label="默认演技", lines=2, max_lines=4,
                                             info="描述语气 情绪和说话方式")
                        with gr.Tabs(elem_id="script-tabs"):
                            with gr.Tab("单句生成"):
                                text = gr.Textbox(label="台词", lines=7, max_lines=14, elem_id="single-text")
                                single_button = gr.Button("生成语音", variant="primary")
                            with gr.Tab("批量生成"):
                                with gr.Accordion("批量添加", open=False):
                                    bulk_text = gr.Textbox(label="台词", lines=3, max_lines=8, elem_id="batch-input")
                                    with gr.Row(elem_id="batch-toolbar"):
                                        split_mode = gr.Radio([("按行", "Line"), ("按段落", "Paragraph")],
                                                              value="Paragraph", label="拆分方式", scale=3)
                                        append_button = gr.Button("加入台词表", interactive=False, scale=1,
                                                                  min_width=125, elem_id="add-lines")
                                queue = gr.Dataframe(value=[[""] * 4 for _ in range(3)], headers=DISPLAY_HEADERS, type="array", datatype="str",
                                                     row_count=3, column_count=(4, "fixed"),
                                                     interactive=True, wrap=True, show_row_numbers=True,
                                                     column_widths=["16%", "44%", "28%", "12%"],
                                                     label="台词表", max_height=350, elem_id="queue-table")
                                with gr.Row():
                                    batch_button = gr.Button("开始批量生成", variant="primary", scale=3)
                                    stop_button = gr.Button("停止后续", scale=1)
                                    clear_button = gr.Button("清空台词", scale=1)
                                continue_on_error = gr.Checkbox(label="出错后继续", value=True)
                                with gr.Accordion("导入 / 导出", open=False):
                                    with gr.Row():
                                        table_upload = gr.File(label="台词文件", file_types=[".txt", ".csv", ".json"], type="filepath")
                                        with gr.Column():
                                            table_browse = gr.UploadButton("选择文件", file_types=[".txt", ".csv", ".json"], type="filepath")
                                            import_table = gr.Button("导入台词", interactive=False)
                                            export_table = gr.Button("导出台词")
                                            table_download = gr.File(label="台词文件", interactive=False)
                        with gr.Accordion("生成参数", open=False):
                            with gr.Row():
                                steps = gr.Slider(10, 80, value=40, step=1, label="采样步数", info="步数越多 生成耗时越长")
                                duration = gr.Slider(0.7, 1.3, value=1.0, step=0.01, label="时长倍率", info="大于 1 延长语音 小于 1 缩短")
                            with gr.Row():
                                speaker_cfg = gr.Slider(1, 8, value=5, step=0.5, label="音色强度", info="越高越贴近音色 过高可能失真")
                                caption_cfg = gr.Slider(1, 8, value=3, step=0.5, label="演技强度", info="越高越强调演技描述")
                                text_cfg = gr.Slider(1, 8, value=3, step=0.5, label="文本强度", info="越高越强调台词内容 过高可能生硬")
                            with gr.Row():
                                seed = gr.Textbox(label="随机种子", value="", info="留空随机 固定数字便于对比")
                                trim_tail = gr.Checkbox(label="裁剪尾部静音", value=False, info="移除生成音频末尾的静音")
                    with gr.Column(scale=1, min_width=340, elem_id="output-column"):
                        status = gr.HTML(status_html())
                        audio = gr.Audio(label="试听", type="filepath", interactive=False,
                                         buttons=["download"], elem_id="audio-preview")
                        preview = gr.Dropdown(label="音频", choices=[], value=None)
                        with gr.Row(elem_id="download-row"):
                            download = gr.DownloadButton("下载当前", interactive=False)
                            archive = gr.DownloadButton("打包下载", interactive=False)
                        with gr.Accordion("运行日志", open=False):
                            log = gr.Textbox(label="日志", lines=12, interactive=False, elem_id="runtime-log")
                with gr.Accordion("生成记录", open=False):
                    results = gr.Dataframe(headers=["序号", "名称 / 台词", "状态", "种子", "时长 / 秒", "错误"],
                                           value=[], type="array", interactive=False, wrap=True, show_label=False,
                                           column_widths=["6%", "30%", "10%", "15%", "9%", "30%"], max_height=350)
            with gr.Tab("音色管理", id="voices"):
                with gr.Row():
                    with gr.Column():
                        upload_voice = gr.File(label="上传音色", file_types=[".safetensors"],
                                                file_count="multiple", type="filepath", elem_id="voice-upload")
                        with gr.Row():
                            voice_browse = gr.UploadButton("选择文件", file_types=[".safetensors"],
                                                           file_count="multiple", type="filepath")
                            import_button = gr.Button("导入文件", variant="primary", interactive=False)
                    with gr.Column():
                        voice_path = path_field("音色文件路径", kind="speaker")
                        import_path_button = gr.Button("从路径导入", interactive=False)
                        path_field("音色目录", value=str(VOICES), interactive=False)
                        library_refresh = gr.Button("刷新音色库")
                library = gr.Dataframe(headers=["音色", "文件名", "大小 / KB"], value=library_rows(),
                                       type="array", interactive=False, label="已导入音色", wrap=True,
                                       column_widths=["25%", "60%", "15%"], elem_id="voice-library")
            with gr.Tab("模型设置", id="models"):
                with gr.Row():
                    with gr.Column(scale=2):
                        checkpoint = path_field("模型", value=DEFAULT_MODEL, kind="checkpoint",
                                                info="本地权重或已缓存的 Hugging Face 模型 ID")
                        codec = path_field("音频编解码器", value=DEFAULT_CODEC, kind="codec",
                                           info="本地 weights.pth 或已缓存的 Hugging Face 模型 ID")
                        lora = path_field("LoRA 目录", info="选择含 adapter_config.json 的目录 留空禁用")
                    with gr.Column(scale=1):
                        device = gr.Dropdown(list_available_runtime_devices(), value=device_default, label="设备")
                        with gr.Row():
                            model_precision = gr.Dropdown(precisions, value="bf16" if "bf16" in precisions else "fp32", label="模型精度",
                                                         info="BF16 更省显存 FP32 数值精度更高")
                            codec_precision = gr.Dropdown(precisions, value="bf16" if "bf16" in precisions else "fp32", label="Codec 精度",
                                                         info="控制音频编解码的计算精度")
                        unload = gr.Button("卸载模型")
                with gr.Accordion("目录", open=False):
                    path_field("运行库", value=str(RUNTIME), interactive=False)
                    path_field("模型缓存", value=str(CACHE), interactive=False)
                    path_field("输出目录", value=str(DATA / "outputs"), interactive=False)

            build_training_tab(demo, ENGINE, voice, library, source, lora, checkpoint)

        settings_inputs = [checkpoint, codec, device, model_precision, codec_precision, source, voice, reference, lora,
                           caption, steps, duration, speaker_cfg, caption_cfg, text_cfg, seed, trim_tail]
        generation_outputs = [results, status, archive, preview, audio, log, download]
        source.change(change_source, source, [voice_group, reference_group, voice_controls], queue=False)
        device.change(change_device, [device, model_precision, codec_precision],
                      [model_precision, codec_precision], queue=False)
        for button in (refresh, library_refresh):
            button.click(refresh_voices, voice, [voice, library], queue=False)
        demo.load(refresh_voices, voice, [voice, library], queue=False)
        reference_browse.upload(lambda selected: selected, reference_browse, reference, queue=False, api_name=False)
        voice_browse.upload(lambda selected: selected, voice_browse, upload_voice, queue=False, api_name=False)
        table_browse.upload(lambda selected: selected, table_browse, table_upload, queue=False, api_name=False)
        upload_voice.change(enable_if_present, upload_voice, import_button, queue=False)
        voice_path.change(enable_if_present, voice_path, import_path_button, queue=False)
        import_button.click(add_voice_file, upload_voice, [voice, library, upload_voice], queue=False, api_name="import_voice_file")
        import_path_button.click(add_voice_path, voice_path, [voice, library, voice_path], queue=False, api_name="import_voice_path")
        bulk_text.change(enable_if_present, bulk_text, append_button, queue=False)
        append_button.click(append_and_clear, [bulk_text, split_mode, queue], [queue, bulk_text], queue=False)
        clear_button.click(lambda: [[""] * 4 for _ in range(3)], outputs=queue, queue=False)
        table_upload.change(enable_if_present, table_upload, import_table, queue=False)
        import_table.click(append_file, [table_upload, split_mode, queue], queue, queue=False)
        export_table.click(download_rows, queue, table_download, queue=False)
        single_button.click(generate_single, [text, *settings_inputs], generation_outputs,
                            concurrency_id="gpu", concurrency_limit=1, api_name="generate_single", show_progress="minimal")
        batch_button.click(generate, [queue, continue_on_error, *settings_inputs], generation_outputs,
                           concurrency_id="gpu", concurrency_limit=1, api_name="generate_batch", show_progress="minimal")
        stop_button.click(stop_batch, queue=False, api_name="stop_batch")
        unload.click(release_model, concurrency_id="gpu", concurrency_limit=1, api_name="release_model")
        preview.input(select_clip, preview, [audio, download], queue=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Irodori TTS")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    print(f"[tts] project={ROOT} data={DATA}", flush=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False,
                inbrowser=not args.no_browser, allowed_paths=[str(DATA)],
                theme=make_theme(), css=CSS)


if __name__ == "__main__":
    main()
