"""我说蓝色真好看有没有懂的"""
from pathlib import Path

import gradio as gr

FONT = ["Segoe UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif"]
MONO = ["Cascadia Code", "Consolas", "Microsoft YaHei", "monospace"]
CSS = (Path(__file__).parent / "assets" / "tts.css").read_text(encoding="utf-8")


def make_theme():
    return gr.themes.Base(primary_hue="blue", neutral_hue="slate", font=FONT, font_mono=MONO).set(
        body_background_fill="#f6f8fb", body_background_fill_dark="#101216",
        body_text_color="#253247", body_text_color_dark="#e0e5ed",
        body_text_size="14px", body_text_color_subdued="#68788d",
        body_text_color_subdued_dark="#919cac",
        background_fill_primary="#ffffff", background_fill_primary_dark="#191d24",
        background_fill_secondary="#f0f3f8", background_fill_secondary_dark="#232832",
        block_background_fill="#ffffff", block_background_fill_dark="#191d24",
        block_border_color="#e0e6ee", block_border_color_dark="#2b323e",
        block_border_width="1px", block_border_width_dark="1px",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_border_width_dark="0px",
        block_label_text_color="#406c9c", block_label_text_color_dark="#8eb6e8",
        block_label_text_weight="500", block_label_text_size="13px",
        block_label_padding="0", block_label_margin="0 0 8px", block_label_radius="0",
        block_title_text_color="#406c9c", block_title_text_color_dark="#8eb6e8",
        block_title_text_weight="500", block_title_text_size="13px",
        block_padding="14px", block_radius="10px", block_shadow="none", block_shadow_dark="none",
        layout_gap="16px", form_gap_width="0px",
        input_background_fill="#f5f7fa", input_background_fill_dark="#232832",
        input_border_color="#e1e6ef", input_border_color_dark="#303846",
        input_border_width="1px", input_border_width_dark="1px",
        input_radius="7px", input_padding="10px 12px", input_text_size="14px",
        input_shadow="none", input_shadow_dark="none",
        input_shadow_focus="0 0 0 2px #5597e525", input_shadow_focus_dark="0 0 0 2px #5597e530",
        border_color_primary="#e0e6ee", border_color_primary_dark="#2b323e",
        border_color_accent="#76a7df", border_color_accent_dark="#568ed1",
        button_primary_background_fill="#3d83d2", button_primary_background_fill_dark="#3d83d2",
        button_primary_background_fill_hover="#3173bd", button_primary_background_fill_hover_dark="#5597e5",
        button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#eef2f7", button_secondary_background_fill_dark="#252c36",
        button_secondary_background_fill_hover="#e2e9f2", button_secondary_background_fill_hover_dark="#303a48",
        button_secondary_text_color="#3c5675", button_secondary_text_color_dark="#bfd0e6",
        button_secondary_border_color="#dfe6ef", button_secondary_border_color_dark="#333e4c",
        button_large_text_size="14px", button_large_text_weight="500", button_large_radius="8px",
        button_large_padding="11px 18px", button_small_text_size="13px",
        button_small_radius="7px", button_small_padding="8px 12px",
        button_primary_shadow="none", button_primary_shadow_dark="none",
        button_secondary_shadow="none", button_secondary_shadow_dark="none",
        checkbox_label_background_fill="#f5f7fa", checkbox_label_background_fill_dark="#232832",
        checkbox_label_background_fill_selected="#e7f0fc", checkbox_label_background_fill_selected_dark="#233a56",
        checkbox_label_text_size="14px", checkbox_label_text_weight="400",
        checkbox_label_padding="9px 12px", checkbox_label_gap="8px",
        checkbox_label_border_color="#e1e6ef", checkbox_label_border_color_dark="#303846",
        checkbox_label_border_color_selected="#7baae0", checkbox_label_border_color_selected_dark="#5488c6",
        table_even_background_fill="#ffffff", table_even_background_fill_dark="#191d24",
        table_odd_background_fill="#f5f7fa", table_odd_background_fill_dark="#1e232c",
        table_border_color="#e0e6ee", table_border_color_dark="#303846",
        table_text_color="#253247", table_text_color_dark="#e0e5ed",
    )

