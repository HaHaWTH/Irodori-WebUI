from html import escape
from pathlib import Path

import gradio as gr


EMOJI_LIST = [
    ("👂", "耳语 咬耳朵"),
    ("😮‍💨", "叹气 呼吸声"),
    ("⏸️", "停顿 沉默"),
    ("🤭", "偷笑 憋笑"),
    ("🥵", "喘息 呻吟"),
    ("📢", "回声 混响"),
    ("😏", "调侃 撒娇"),
    ("🥺", "颤抖 怯生生"),
    ("🌬️", "气喘 急促呼吸"),
    ("😮", "倒吸气 惊"),
    ("👅", "舔 咀嚼水声"),
    ("💋", "咂嘴 唇音"),
    ("🫶", "温柔"),
    ("😭", "哭腔 呜咽"),
    ("😱", "尖叫"),
    ("😪", "困倦 慵懒"),
    ("😴", "梦话 打鼾"),
    ("⏩", "快语速 急"),
    ("📞", "电话音 听筒"),
    ("🐢", "慢速"),
    ("🥤", "吞咽"),
    ("🤧", "咳嗽 抽鼻"),
    ("😒", "咂舌 不屑"),
    ("😰", "慌张 紧张 结巴"),
    ("😆", "开心 欢喜"),
    ("💥", "气势 有力"),
    ("😠", "生气 不满"),
    ("😲", "惊讶 感叹"),
    ("🥱", "打哈欠"),
    ("😖", "痛苦"),
    ("😟", "担心 不安"),
    ("🫣", "害羞"),
    ("🙄", "无语 翻白眼"),
    ("😊", "愉快"),
    ("😎", "得意 自信"),
    ("👌", "附和 点头"),
    ("🙏", "恳求 拜托"),
    ("🥴", "醉醺醺"),
    ("🎵", "哼歌"),
    ("🤐", "闷声 捂嘴"),
    ("😌", "安心 满足"),
    ("🤔", "疑问"),
    ("💪", "用力 坚定"),
    ("👃", "嗅闻"),
    ("📖", "朗读 旁白"),
]


def build_emoji_picker():
    buttons = "".join(
        f'<button type="button" data-emoji="{escape(emoji, quote=True)}" '
        f'title="{escape(meaning, quote=True)}" aria-label="{escape(meaning, quote=True)}">'
        f'{emoji}</button>'
        for emoji, meaning in EMOJI_LIST
    )
    return gr.HTML(
        '<details class="emoji-picker"><summary>表情</summary>'
        f'<div class="emoji-grid">{buttons}</div>'
        '<p class="emoji-status" role="status" hidden></p></details>',
        elem_id="emoji-picker", apply_default_css=False,
        js_on_load=(Path(__file__).parent / "assets" / "emoji.js").read_text(encoding="utf-8"),
    )
