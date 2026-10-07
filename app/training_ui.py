from __future__ import annotations

from pathlib import Path

import gradio as gr
import yaml

from tts_bootstrap import DATA
from tts_engine import DEFAULT_CODEC, DEFAULT_MODEL, import_voice, voice_choices
from tts_paths import path_field
from tts_training import JOBS, TrainingSettings, artifacts, log_tail, read_samples, start_preparation, start_training, training_config
from irodori_tts.inference_runtime import default_runtime_device, list_available_runtime_devices, list_available_runtime_precisions


def read_training_settings(checkpoint, manifest, mode, device, precision, name, output_root,
                           steps, batch_size, accumulation, learning_rate, save_every,
                           gradient_checkpointing, max_frames, tokens, lora_rank, lora_alpha,
                           lora_target, continuation, seed):
    return TrainingSettings(checkpoint=checkpoint, manifest=manifest, mode=mode, device=device,
                            precision=precision, name=name, output_root=output_root, steps=int(steps),
                            batch_size=int(batch_size), accumulation=int(accumulation),
                            learning_rate=float(learning_rate), save_every=int(save_every),
                            gradient_checkpointing=gradient_checkpointing, max_frames=int(max_frames),
                            tokens=int(tokens), lora_rank=int(lora_rank), lora_alpha=int(lora_alpha),
                            lora_target=lora_target, continuation=continuation, seed=int(seed))


def check_data(directory, table, speaker, caption, max_seconds):
    try:
        samples = read_samples(directory, table, speaker, caption, float(max_seconds))
        rows = [[Path(item["audio"]).name, item["text"], item["caption"], item["speaker_id"], item["seconds"]]
                for item in samples]
        return rows, f"{len(samples)} 条 · {sum(item['seconds'] for item in samples):.1f} 秒"
    except Exception as exc:
        raise gr.Error(f"Dataset check failed: {exc}") from exc


def preview_training(*values):
    try:
        _, _, config = training_config(read_training_settings(*values))
        return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    except Exception as exc:
        raise gr.Error(f"Training configuration failed: {exc}") from exc


def change_mode(mode):
    speaker = mode == "speaker"
    return (gr.update(visible=speaker), gr.update(visible=not speaker),
            gr.update(value=.01 if speaker else .0001),
            gr.update(label="继续训练音色" if speaker else "继续训练检查点", value="",
                      info="可选 加载已有音色 步数和优化器重新开始" if speaker
                      else "可选 选择含 trainer_state.pt 的目录 恢复训练状态"))


def change_training_device(device, precision):
    choices = list_available_runtime_precisions(device)
    return gr.update(choices=choices, value=precision if precision in choices else "fp32")


def job_started(job):
    return (job.id, gr.update(active=True), "预处理中" if job.kind == "prepare" else "训练中",
            "Starting worker...", str(job.directory), gr.update(choices=[], value=None),
            gr.update(interactive=False), gr.update(interactive=True))


def poll_job(job_id):
    if not job_id:
        return ("待开始", "", gr.update(choices=[], value=None), gr.update(interactive=False),
                gr.update(active=False), gr.skip(), gr.update(interactive=False))
    job = JOBS.get(job_id)
    running = job.status == "Running"
    result = [] if running else artifacts(job)
    manifest = result[0] if result and job.kind == "prepare" and job.status == "Done" else gr.skip()
    state = {"Running": "运行中", "Done": "已完成", "Failed": "失败", "Stopped": "已停止"}[job.status]
    if job.status == "Failed":
        state += f" · exit {job.returncode}"
    return (state, log_tail(job.log).replace("\r", "\n"),
            gr.update(choices=[(Path(path).name, path) for path in result], value=result[0] if result else None),
            gr.update(interactive=bool(result) and job.kind == "train"), gr.update(active=running), manifest,
            gr.update(interactive=running))


def restore_job():
    job = JOBS.latest()
    return (job.id, str(job.directory)) if job else ("", "")


def stop_job(job_id):
    if not job_id:
        gr.Info("No active training job.")
        return
    JOBS.stop(job_id)
    gr.Info("Stop requested. Previously saved checkpoints are retained.")


def apply_result(job_id, result):
    try:
        job = JOBS.get(job_id)
        if job.status == "Running":
            raise ValueError("Wait for training to finish or stop it before applying a checkpoint.")
        if not result or result not in artifacts(job):
            raise ValueError("Select a completed training checkpoint.")
        if job.mode == "speaker":
            voices = DATA / "voices"
            path = import_voice(result, voices, name=job.directory.name + ".speaker.safetensors")
            rows = [[label, Path(value).name, round(Path(value).stat().st_size / 1024, 1)]
                    for label, value in voice_choices(voices)]
            return gr.update(choices=voice_choices(voices), value=path), rows, "Speaker file", "", job.checkpoint
        return gr.skip(), gr.skip(), gr.skip(), result, job.checkpoint
    except Exception as exc:
        raise gr.Error(f"Checkpoint import failed: {exc}") from exc


def build_training_tab(demo, engine, voice, library, source, inference_lora, inference_checkpoint):
    device_default = default_runtime_device()
    precision_choices = list_available_runtime_precisions(device_default)
    with gr.Tab("训练", id="training"):
        with gr.Tabs():
            with gr.Tab("数据准备"):
                data_directory = path_field("音频目录", info="支持 WAV FLAC OGG MP3 可包含子目录")
                data_table = path_field("标注 CSV", kind="csv", info="可选 留空时读取音频同名 TXT")
                with gr.Row():
                    data_name = gr.Textbox(label="数据集名称", value="dataset", info="用于区分预处理输出")
                    data_speaker = gr.Textbox(label="说话人", value="speaker", info="同一声线使用同一标识")
                    data_caption = gr.Textbox(label="默认演技", value="", info="未单独标注演技时使用")
                with gr.Accordion("预处理参数", open=False):
                    data_codec = path_field("音频编解码器", value=DEFAULT_CODEC, kind="codec",
                                            info="使用与基础模型匹配的 Codec")
                    with gr.Row():
                        data_max_seconds = gr.Number(label="最长音频 / 秒", value=30, minimum=.1,
                                                     info="超长音频会报错 需先切分音频和文本")
                        data_precision = gr.Dropdown(precision_choices, value="fp32", label="Codec 精度",
                                                     info="FP32 数值精度更高 BF16 更省显存")
                        data_device = gr.Dropdown(list_available_runtime_devices(), value=device_default, label="设备")
                    data_output = path_field("数据输出目录", value=str(DATA / "training" / "datasets"),
                                             info="每次预处理创建独立子目录")
                with gr.Row():
                    data_check = gr.Button("检查数据")
                    data_prepare = gr.Button("预处理", variant="primary")
                data_summary = gr.Textbox(label="数据统计", interactive=False)
                data_preview = gr.Dataframe(headers=["音频", "文本", "演技", "说话人", "时长 / 秒"],
                                             value=[], type="array", interactive=False, wrap=True, max_height=300,
                                             label="数据清单")
            with gr.Tab("训练设置"):
                with gr.Row():
                    name = gr.Textbox(label="训练名称", value="voice", info="用于命名本次训练输出")
                    mode = gr.Radio([("音色训练", "speaker"), ("LoRA 微调", "lora")], value="speaker", label="训练方式",
                                    info="音色训练只优化音色向量 LoRA 微调模型")
                checkpoint = path_field("基础模型", value=DEFAULT_MODEL, kind="checkpoint",
                                        info="本地权重或已缓存的 Hugging Face 模型 ID")
                manifest = path_field("训练清单", kind="manifest", info="选择预处理生成的 manifest.jsonl")
                with gr.Row():
                    device = gr.Dropdown(list_available_runtime_devices(), value=device_default, label="训练设备")
                    precision = gr.Dropdown(precision_choices, value="bf16" if "bf16" in precision_choices else "fp32", label="训练精度",
                                           info="控制前向计算 权重和优化器仍为 FP32")
                    steps = gr.Number(label="训练步数", value=3000, precision=0, minimum=1, info="优化器更新的总次数")
                with gr.Row():
                    batch_size = gr.Number(label="批次大小", value=1, precision=0, minimum=1, info="每个批次的样本数 越大越占显存")
                    accumulation = gr.Number(label="梯度累积", value=4, precision=0, minimum=1,
                                             info="累计多少批次再更新 有效批次大小 = 批次大小 × 累积次数")
                with gr.Row():
                    learning_rate = gr.Number(label="学习率", value=.01, minimum=0, info="每次更新幅度 过大容易不稳定")
                    save_every = gr.Number(label="保存间隔", value=250, precision=0, minimum=1,
                                           info="每隔多少训练步保存一次检查点")
                with gr.Group() as speaker_options:
                    tokens = gr.Number(label="音色 Token 数", value=16, precision=0, minimum=1,
                                       info="可学习的音色向量数 继续训练时需匹配")
                with gr.Group(visible=False) as lora_options:
                    with gr.Row():
                        lora_rank = gr.Number(label="LoRA Rank", value=8, precision=0, minimum=1,
                                             info="适配器容量 越大训练参数越多")
                        lora_alpha = gr.Number(label="LoRA Alpha", value=16, precision=0, minimum=1,
                                              info="适配器缩放系数为 Alpha / Rank")
                    lora_target = gr.Dropdown([("扩散模型注意力", "diffusion_attn"),
                                              ("扩散模型注意力与前馈层", "diffusion_attn_mlp"),
                                              ("全部注意力", "all_attn"), ("全部线性层", "all_linear")],
                                              value="diffusion_attn", label="LoRA 模块",
                                              info="选择参与训练的层 范围越广越占显存")
                with gr.Accordion("高级设置", open=False):
                    gradient_checkpointing = gr.Checkbox(label="梯度检查点", value=True,
                                                         info="以额外计算换取更低显存占用")
                    with gr.Row():
                        max_frames = gr.Number(label="最大潜在帧数", value=750, precision=0, minimum=1,
                                               info="单条样本的帧数上限 超出会报错 越长越占显存")
                        seed = gr.Number(label="随机种子", value=0, precision=0, minimum=0,
                                         info="控制训练随机性 0 也是固定种子")
                    continuation = path_field("继续训练音色", kind="speaker", mode=mode,
                                              info="可选 加载已有音色 步数和优化器重新开始")
                    output_root = path_field("训练输出目录", value=str(DATA / "training" / "runs"),
                                             info="每次训练创建独立子目录 保存配置 日志和检查点")
                with gr.Row():
                    config_button = gr.Button("预览配置")
                    train_button = gr.Button("开始训练", variant="primary")
                with gr.Accordion("训练配置", open=False):
                    config_preview = gr.Code(language="yaml", label="YAML", interactive=False)
        with gr.Row():
            status = gr.Textbox(label="任务状态", value="待开始", interactive=False, scale=3)
            stop = gr.Button("停止任务", interactive=False, scale=1)
        output_directory = path_field("当前输出目录", interactive=False)
        logs = gr.Textbox(label="训练日志", lines=12, max_lines=18, interactive=False)
        with gr.Row():
            result = gr.Dropdown(label="训练结果", choices=[], value=None, scale=4)
            use_result = gr.Button("用于合成", interactive=False, scale=1)
        job_id = gr.State("")
        timer = gr.Timer(1, active=False)

    training_inputs = [checkpoint, manifest, mode, device, precision, name, output_root,
                       steps, batch_size, accumulation, learning_rate, save_every,
                       gradient_checkpointing, max_frames, tokens, lora_rank, lora_alpha,
                       lora_target, continuation, seed]
    preparation_inputs = [data_name, data_directory, data_table, data_speaker, data_caption, data_max_seconds,
                          data_codec, data_device, data_precision, data_output]
    started_outputs = [job_id, timer, status, logs, output_directory, result, use_result, stop]
    poll_outputs = [status, logs, result, use_result, timer, manifest, stop]

    def prepare(*values):
        try:
            return job_started(start_preparation(*values, release_model=engine.release))
        except Exception as exc:
            raise gr.Error(f"Preprocessing could not start: {exc}") from exc

    def train(*values):
        try:
            return job_started(start_training(read_training_settings(*values), engine.release))
        except Exception as exc:
            raise gr.Error(f"Training could not start: {exc}") from exc

    data_check.click(check_data, [data_directory, data_table, data_speaker, data_caption, data_max_seconds],
                     [data_preview, data_summary], queue=False, api_name="check_training_data")
    data_prepare.click(prepare, preparation_inputs, started_outputs, concurrency_id="gpu", concurrency_limit=1,
                       api_name="prepare_training_data")
    train_button.click(train, training_inputs, started_outputs, concurrency_id="gpu", concurrency_limit=1,
                       api_name="start_training")
    config_button.click(preview_training, training_inputs, config_preview, queue=False, api_name="preview_training_config")
    mode.change(change_mode, mode, [speaker_options, lora_options, learning_rate, continuation], queue=False)
    device.change(change_training_device, [device, precision], precision, queue=False)
    data_device.change(change_training_device, [data_device, data_precision], data_precision, queue=False)
    timer.tick(poll_job, job_id, poll_outputs, queue=False, api_name=False)
    stop.click(stop_job, job_id, queue=False, api_name="stop_training")
    use_result.click(apply_result, [job_id, result], [voice, library, source, inference_lora, inference_checkpoint],
                     concurrency_id="gpu", concurrency_limit=1, api_name="apply_training_result")
    demo.load(restore_job, outputs=[job_id, output_directory], queue=False).then(poll_job, job_id, poll_outputs, queue=False)
