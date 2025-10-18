import gradio as gr
from functools import partial
from .inference import analyze_video

def create_ui(detectors_info, classes, method_names, img_size, det_thr, cli_args, model_labels):
    def _auto_thr_from_selected(selected):
        thrs = [detectors_info[i][6] for i, label in enumerate(model_labels) if (not selected or label in selected)]
        return float(sum(thrs) / len(thrs)) if thrs else float(det_thr)

    with gr.Blocks(title="Deepfake Detect (UI)") as demo:
        gr.Markdown("## Deepfake Detect — Phát hiện video giả mạo (Deepfake)")

        with gr.Row():
            with gr.Column():
                face_crop = gr.Checkbox(label="Face crop", value=True)
                auto_thr = gr.Checkbox(label="Dùng ngưỡng từ mô hình (tự động)", value=True)
                thr = gr.Slider(0.0, 0.99, value=_auto_thr_from_selected(None), step=0.005,
                                label="Ngưỡng FAKE (p_fake ≥ thr ⇒ FAKE)", interactive=False)
                tta = gr.Slider(1, 4, value=2, step=1, label="TTA (1..4)")
                thick = gr.Slider(1, 8, value=3, step=1, label="Độ dày khung")
            with gr.Column():
                enable_filters = gr.Checkbox(label="Bật filter (unsharp/deblock/denoise)",
                                             value=bool(cli_args.get('enable_filters', True)))
                gr.Markdown(f"**img={img_size}**  \n**Classes**: {classes}  \n**Methods**: {', '.join(method_names)}")

        gr.Markdown("### Chọn mô hình (phải bật ≥ 1)")
        model_selector = gr.CheckboxGroup(choices=model_labels, value=model_labels, label="Models",
                                          interactive=(len(model_labels) > 1))
        if len(model_labels) == 1:
            gr.Markdown("*Chỉ có 1 mô hình → luôn bật (không thể tắt).*")

        def _on_models_change(selected, auto_flag):
            if auto_flag:
                return gr.update(value=_auto_thr_from_selected(selected))
            return gr.update()
        model_selector.change(_on_models_change, inputs=[model_selector, auto_thr], outputs=[thr])

        def _on_auto_change(auto_flag, selected):
            return (gr.update(interactive=not auto_flag),
                    gr.update(value=_auto_thr_from_selected(selected) if auto_flag else gr.update()))
        auto_thr.change(_on_auto_change, inputs=[auto_thr, model_selector], outputs=[thr, thr])

        gr.Markdown("### Detect video")
        with gr.Row():
            in_vid = gr.Video(label="Video nguồn")
            out_vid = gr.Video(label="Video đã phân tích")
        vid_text = gr.Textbox(label="Nhật ký", interactive=False)
        fr_bar_vid = gr.HTML(label="Tỉ lệ Fake/Real (video)")
        method_df_vid = gr.Dataframe(headers=["Method", "%"], datatype=["str", "number"], interactive=False,
                                     label="Tỉ lệ theo phương pháp (chỉ hiển thị nếu FAKE)")

        _vid_wrap_partial = partial(
            _vid_wrap,
            detectors_info=detectors_info,
            method_names=method_names,
            det_thr=float(det_thr),
            model_labels=model_labels
        )
        btn_vid = gr.Button("Phân tích Video")
        btn_vid.click(
            _vid_wrap_partial,
            inputs=[in_vid, face_crop, auto_thr, thr, tta, thick, enable_filters, model_selector],
            outputs=[out_vid, vid_text, fr_bar_vid, method_df_vid]
        )
    return demo

def _vid_wrap(vid_path, fc, auto_thr, thr, tta, thick, enable_filters, selected_labels,
              detectors_info, method_names, det_thr, model_labels):
    if not vid_path:
        return None, "Chưa chọn video.", "", []

    if not selected_labels:
        selected_labels = [model_labels[0]]
    active = [detectors_info[i] for i, label in enumerate(model_labels) if label in selected_labels]

    override_thr = None if auto_thr else float(thr)

    out_path, verdict, fr_bar_html, method_rows = analyze_video(
        vid_path, active, method_names, det_thr,
        use_face_crop=bool(fc),
        override_thr=override_thr,
        tta=int(tta),
        box_thickness=int(thick),
        enable_filters=bool(enable_filters),
        saliency_density=0.02,
        saliency_mode="method"
    )
    return out_path, verdict, fr_bar_html, method_rows
