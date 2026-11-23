"""Bilingual (Chinese / English) UI localization and parameter label translation."""
from __future__ import annotations

_CURRENT_LANG = "en"

PARAM_NAMES_ZH: dict[str, str] = {
    "Exposure": "曝光 Exposure",
    "Contrast": "对比度 Contrast",
    "Highlights": "高光 Highlights",
    "Shadows": "阴影 Shadows",
    "Whites": "白色 Whites",
    "Blacks": "黑色 Blacks",
    "Vibrance": "自然饱和度 Vibrance",
    "Saturation": "饱和度 Saturation",
    "Warmth": "色温 Warmth",
    "Tint": "色调 Tint",
    "Feather": "羽化 Feather",
    "Angle": "角度 Angle",
    "Brush size": "画笔大小 Brush size",
    "Brush feather": "画笔羽化 Brush feather",
    "Luma min": "亮度下限 Luma min",
    "Luma max": "亮度上限 Luma max",
    "Range feather": "范围羽化 Range feather",
    "Spot size": "污点大小 Spot size",
    "Opacity": "不透明度 Opacity",
    "Distortion": "畸变 Distortion",
    "Vignetting": "暗角 Vignetting",
    "Sharpening": "锐化 Sharpening",
    "Edge masking": "边缘蒙版 Edge masking",
    "Luminance NR": "明度降噪 Luminance NR",
    "Color NR": "彩色降噪 Color NR",
    "Hue": "色相 Hue",
    "Lightness": "明度 Lightness",
    "Luminance": "明度 Luminance",
    "Blend": "混合 Blend",
    "Balance": "平衡 Balance",
}


def get_language() -> str:
    return _CURRENT_LANG


def set_global_language(lang: str) -> str:
    global _CURRENT_LANG
    _CURRENT_LANG = "zh" if str(lang).lower().startswith("zh") else "en"
    return _CURRENT_LANG


def translate_parameter_name(name: str, lang: str | None = None) -> str:
    active = lang or _CURRENT_LANG
    if active == "zh":
        return PARAM_NAMES_ZH.get(name, name)
    return name


def tr(en_text: str, zh_text: str, lang: str | None = None) -> str:
    active = lang or _CURRENT_LANG
    return zh_text if active == "zh" else en_text


def apply_workspace_language(window, lang: str) -> None:
    """Switch all MainWindow controls between English ('en') and Chinese ('zh')."""
    active = set_global_language(lang)
    zh = active == "zh"

    if hasattr(window, "lang_button"):
        window.lang_button.blockSignals(True)
        window.lang_button.setChecked(zh)
        window.lang_button.setText("EN / 中文" if zh else "中文 / EN")
        window.lang_button.setToolTip(
            "切换界面与教练语言为英文 (Switch UI & Coach to English)"
            if zh
            else "Switch UI & Coach language between Chinese and English (切换中英双语)"
        )
        window.lang_button.blockSignals(False)

    # Top Toolbar
    window.open_button.setText("打开" if zh else "Open")
    if hasattr(window, "open_raw_action"):
        window.open_raw_action.setText("打开 RAW 照片…" if zh else "Open RAW…")
    if hasattr(window, "open_project_button"):
        window.open_project_button.setText("打开工程…" if zh else "Open project…")
    window.undo_button.setText("撤销" if zh else "Undo")
    window.redo_button.setText("重做" if zh else "Redo")
    window.save_project_button.setText("保存" if zh else "Save")
    window.photo_info_button.setText("信息" if zh else "Info")
    window.export_button.setText("导出" if zh else "Export")
    window.more_button.setText("更多" if zh else "More")
    if not window.store.current:
        window.filename.setText("未打开照片" if zh else "No photo open")
    if hasattr(window, "unsaved_label"):
        window.unsaved_label.setText("未保存修改" if zh else "Unsaved changes")

    # Canvas Toolbar
    window.fit_button.setText("适应窗口" if zh else "Fit")
    window.clipping.setText("溢出警告" if zh else "Clipping")
    window.compare.setText("原片基准" if zh else "Baseline")
    window.edited_button.setText("编辑效果" if zh else "Edited")
    window.composition_button.setText("裁剪 / 拉直" if zh else "Crop / Straighten")
    if hasattr(window, "canvas_hint_label"):
        window.canvas_hint_label.setText(
            "Ctrl + 滚轮：缩放 · 空格 + 拖动：平移 · 双击：适应窗口 / 100%"
            if zh
            else "Ctrl + wheel: zoom · Space + drag: pan · Double-click: Fit / 100%"
        )

    # Tabs & Sections
    window.tabs.setTabText(0, "调色编辑" if zh else "Edit")
    window.tabs.setTabText(1, "历史记录" if zh else "History")
    window.base_rendering_label.setText("基础渲染" if zh else "Base rendering")
    for index, title in enumerate(("自然 / 拍摄时白平衡", "中性 / 平直", "旧版") if zh else ("Natural / As Shot", "Neutral / Flat", "Legacy")):
        window.base_rendering.setItemText(index, title)
    window.light_section.toggle.setText("光效 Light" if zh else "Light")
    window.crop_section.toggle.setText("裁剪与拉直 Crop" if zh else "Crop and straighten")
    window.lens_section.toggle.setText("镜头校正 Lens" if zh else "Lens corrections")
    window.detail_section.toggle.setText("细节与降噪 Detail" if zh else "Detail")
    window.local_section.toggle.setText("局部蒙版与光束 Local" if zh else "Local adjustments")
    if hasattr(window, "spots_section"):
        window.spots_section.toggle.setText("污点修复 Spots" if zh else "Spot removal")
    window.color_section.toggle.setText("色彩 Color" if zh else "Color")
    window.curve_section.toggle.setText("曲线 Curve" if zh else "Curve")

    # Crop & Footer buttons
    window.crop_open_button.setText("裁剪与拉直…" if zh else "Crop and straighten…")
    window.crop_reset_button.setText("重置" if zh else "Reset")
    if hasattr(window, "wb_label"):
        window.wb_label.setText("白平衡" if zh else "White balance")
    window.advanced_wb.setText("高级 RGB 增益" if zh else "Advanced RGB gains")
    window.apply_wb.setText("应用白平衡" if zh else "Apply white balance")
    if not window.pick_wb.isChecked():
        window.pick_wb.setText("吸取中性灰白平衡" if zh else "Pick neutral white balance")
    window.curve_label.setText("RGB 共享曲线" if zh else "Shared RGB curve")
    window.reset_curve_button.setText("重置曲线" if zh else "Reset curve")
    window.reset_button.setText("重置全部修改" if zh else "Reset edits")
    if hasattr(window, "help_button"):
        window.help_button.setText("编辑帮助" if zh else "Editing help")
    if hasattr(window, "history_note"):
        window.history_note.setText(
            "已提交的操作 · 完整审计记录保存在工程中"
            if zh
            else "Committed actions · Complete audit is saved in the project"
        )
    if hasattr(window, "snapshots_panel"):
        window.snapshots_panel.set_language(active)
    if hasattr(window, "_refresh_view_notice"):
        window._refresh_view_notice()
    if window.store.current and window.store.current.compare_snapshot is not None:
        snap_name = window.store.current.compare_snapshot.name
        window.status.setText(f"A/B 对比：{snap_name}" if zh else f"Comparing A/B: {snap_name}")

    # Sub-editors
    if hasattr(window.detail_editor, "note"):
        window.detail_editor.note.setText(
            "在 100% 视图下检查 · 明度锐化与彩色降噪"
            if zh
            else "Inspect at 100% · Luminance USM and color NR"
        )
    window.detail_editor.preset.setText("Camera Raw 预设" if zh else "Camera Raw preset")
    window.detail_editor.reset.setText("重置细节" if zh else "Reset detail")
    window.detail_editor.set_values(window.detail_editor.values)

    if hasattr(window.lens_editor, "note"):
        window.lens_editor.note.setText("手动几何畸变与暗角校正" if zh else "Manual correction")
    window.lens_editor.reset.setText("重置镜头校正" if zh else "Reset lens corrections")
    window.lens_editor.set_values(window.lens_editor.values)

    window.color_mixer.toggle.setText("混色器 Color Mixer" if zh else "Color Mixer")
    window.color_mixer.reset_color.setText("重置当前颜色" if zh else "Reset selected color")
    window.color_mixer.reset_all.setText("重置混色器" if zh else "Reset Color Mixer")
    window.color_mixer.set_values(window.color_mixer.values)

    cg = window.color_grading
    cg.toggle.setText("色彩分级 Color Grading" if zh else "Color Grading")
    for zone_key, zh_label, en_label in (
        ("shadows", "阴影", "Shadows"),
        ("midtones", "中间调", "Midtones"),
        ("highlights", "高光", "Highlights"),
    ):
        cg.zone_buttons[zone_key].setText(zh_label if zh else en_label)
    cg.reset_zone.setText("重置当前分区" if zh else "Reset zone")
    cg.reset_all.setText("重置色彩分级" if zh else "Reset Color Grading")
    cg.set_values(cg.values)

    # Local Gradients Editor
    ge = window.local_gradients
    ge.new_mask_label.setText("新建蒙版" if zh else "New mask")
    for idx, shape_label in enumerate(("线性 Linear", "径向 Radial", "画笔 Brush") if zh else ("Linear", "Radial", "Brush")):
        ge.new_shape.setItemText(idx, shape_label)
    ge.name.setPlaceholderText("蒙版名称" if zh else "Gradient name")
    ge.add.setText("添加蒙版" if zh else "Add mask")
    ge.done.setText("完成" if zh else "Done")
    ge.brush_paint.setText("涂抹" if zh else "Paint")
    ge.brush_erase.setText("擦除" if zh else "Erase")
    ge.enabled.setText("启用" if zh else "Enabled")
    ge.inverted.setText("反向" if zh else "Invert")
    ge.edge_aware.setText("边缘贴合" if zh else "Edge-aware")
    ge.redraw.setText("重绘蒙版" if zh else "Redraw mask")
    ge.delete.setText("删除蒙版" if zh else "Delete mask")
    ge.reset.setText("重置蒙版参数" if zh else "Reset adjustments")
    ge.edit_on_photo.setText("在画面上编辑" if zh else "Edit on photo")
    ge.show_mask.setText("显示蒙版范围" if zh else "Show selected mask")
    ge.set_values(ge.values)
    window.local_controller.sync()

    if hasattr(window, "spots_editor"):
        window.spots_editor.set_language(active)
        window.spots_controller.sync()

    # Refresh ParameterLabels on MainWindow
    for label in (window.exposure_label, *window.balance_labels, *window.adjustment_labels.values()):
        label.refresh_language()

    # Coach controls share their state, while labels follow the window language.
    window.coach_title_label.setText("AI 教练" if zh else "Coach")
    window.coach_menu_button.setText("更多" if zh else "More")
    window.edit_goal_action.setText("编辑目标" if zh else "Edit goal")
    window.coach_usage_action.setText("用量与诊断" if zh else "Usage and diagnostics")
    window.coach_selection.setWindowTitle("选择教练" if zh else "Choose coach")
    window.coach_details.setWindowTitle("教练用量与诊断" if zh else "Coach usage and diagnostics")
    window.coach_selection_close.setText("关闭" if zh else "Close")
    window.coach_details_close.setText("关闭" if zh else "Close")
    window.coach_provider_label.setText("连接" if zh else "Connection")
    window.coach_model_label.setText("模型" if zh else "Model")
    window.coach_connection_button.setText("云端连接" if zh else "Cloud connections")
    window.goal_form_label.setText("编辑目标" if zh else "Editing goal")
    window.goal.setPlaceholderText("希望这张照片表达什么？" if zh else "What would you like this photo to express?")
    window.question.setPlaceholderText("描述问题，留空可评估当前照片。Ctrl+Enter 分析。" if zh else
                                       "Ask about this photo, or leave blank for a review. Ctrl+Enter to analyze.")
    window.question._resize_lines()
    window.coach_questions_button.setText("常用问题" if zh else "Common questions")
    questions = (
        ("下一步建议", "结合当前画面与参数，下一步最该调整什么？什么时候该停下？"),
        ("曲线指导", "这张照片该如何使用 RGB 共享曲线？有充分依据时给出一个可撤销的试调建议。"),
        ("局部布光", "评估画面可见光线，仅在位置和目的有图像依据时建议局部调整。"),
        ("效果复盘：调整有效吗？", "结合上一轮教练建议和我的编辑目标，这次调整有帮助吗？请区分客观画面变化与主观偏好，并说明是否该停下。"),
    ) if zh else (
        ("Next step", "What should I adjust next, and when should I stop?"),
        ("Curve help", "How should I use the shared RGB curve on this photo? Suggest a reversible trial only if the evidence supports it."),
        ("Local lighting", "Assess the visible light and suggest a local adjustment only if its location and purpose are supported by the photo."),
        ("Follow-up: Did it help?", "Did my adjustment help relative to your previous advice and my editing goal? Separate objective preview changes from preference, and tell me if I should stop."),
    )
    for action, (title, question) in zip(window.coach_question_actions, questions):
        action.setText(title)
        action.setProperty("question_text", question)
    window.coach_light_menu_button.setText("布光预设" if zh else "Lighting presets")
    titles = ("山体光束与环境压暗", "主体聚光与环境压暗", "顶部光晕", "侧向窗户光", "逆光轮廓", "天空渐变压暗", "前景补光")
    from .coach_actions import LIGHTING_RECIPE_META
    for action, title, (_, english, _) in zip(window.coach_light_apply_actions, titles, LIGHTING_RECIPE_META):
        action.setText(title if zh else english)
    window.pause_coach.setText(("恢复教练" if window.coach.paused else "暂停教练") if zh else
                               ("Resume coach" if window.coach.paused else "Pause coach"))
    if hasattr(window, "style_memory_enabled"):
        window.style_memory_enabled.setText("在本地记住已采纳的风格偏好" if zh else "Remember applied style preferences locally")
    if hasattr(window, "clear_style_memory_button"):
        window.clear_style_memory_button.setText("清空风格记忆" if zh else "Clear style memory")
    window.coach_response.set_language(active)
    if getattr(window, "_local_dialog", None) is not None and hasattr(window._local_dialog, "set_language"):
        window._local_dialog.set_language(active)
    window._coach_toggle_text()
    window._update_coach_status()
