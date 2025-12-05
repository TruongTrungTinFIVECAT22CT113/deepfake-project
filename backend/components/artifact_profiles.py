# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, TypedDict


class ArtifactProfile(TypedDict):
    summary: str
    artifacts: List[Tuple[str, str]]  # (artifact_name, description)


# Hồ sơ lỗi cho từng phương pháp fake
ARTIFACT_PROFILES: Dict[str, ArtifactProfile] = {
    "Audio2Animation": {
        "summary": (
            "Audio2Animation sinh biểu cảm khuôn mặt từ giọng nói. "
            "Thường lộ lỗi ở vùng miệng và chuyển động đầu, vì phải "
            "suy ra chuyển động từ âm thanh chứ không dựa vào cảnh thật."
        ),
        "artifacts": [
            (
                "Mouth Artifact",
                "Miệng méo nhẹ, răng không rõ hoặc khẩu hình không hoàn toàn khớp chuyển động.",
            ),
            (
                "Lighting Error",
                "Ánh sáng trên mặt khá cố định, không thay đổi đúng với chuyển động đầu và ánh sáng môi trường.",
            ),
            (
                "Geometry Distortion",
                "Khi head shake hoặc quay nhanh, vùng má / cằm bị kéo giãn hoặc biến dạng nhẹ.",
            ),
        ],
    },

    "Deepfakes": {
        "summary": (
            "Deepfakes cổ điển (autoencoder) thường tạo mask (mặt nạ) khuôn mặt rồi dán lên video gốc. "
            "Điểm yếu rõ nhất là viền mặt, lệch tông màu da và da mịn bất thường."
        ),
        "artifacts": [
            (
                "Boundary Error",
                "Đường viền mặt mờ/nhòe, có cảm giác mặt được dán lên khung hình (mask line).",
            ),
            (
                "Color Mismatch",
                "Tông màu da trên mặt khác với cổ hoặc vùng da xung quanh (sáng hơn, hồng hơn hoặc vàng hơn).",
            ),
            (
                "Texture Abnormality",
                "Da mặt mịn như nhựa, mất chi tiết lỗ chân lông, nếp nhăn so với vùng cổ/trán.",
            ),
        ],
    },

    "Face2Face": {
        "summary": (
            "Face2Face điều khiển biểu cảm khuôn mặt dựa trên một actor (diễn viên) khác. "
            "Thường lộ lỗi khi miệng hoạt động mạnh và khi ánh sáng / bóng trên mặt không khớp với chuyển động."
        ),
        "artifacts": [
            (
                "Mouth Artifact",
                "Khi nhân vật cười, nói lớn hoặc phát âm phức tạp, vùng miệng dễ bị méo, môi hoặc răng trông không tự nhiên.",
            ),
            (
                "Lighting Error",
                "Khi nhân vật quay đầu, ánh sáng trên mặt gần như không đổi, không ăn theo hướng nguồn sáng của cảnh.",
            ),
            (
                "Geometry Distortion",
                "Một số khung hình có méo nhẹ vùng cằm/ má khi biểu cảm thay đổi nhanh.",
            ),
        ],
    },

    "FaceShifter": {
        "summary": (
            "FaceShifter là GAN chất lượng cao, che giấu lỗi tốt hơn nhưng vẫn để lộ ở vùng mắt "
            "và texture da khi quan sát kỹ."
        ),
        "artifacts": [
            (
                "Eye Artifact",
                "Độ sắc nét/độ sáng hai mắt không đồng đều, highlight trong mắt không khớp với ánh sáng chung.",
            ),
            (
                "Texture Abnormality",
                "Da thiếu các chi tiết rất nhỏ (micro-detail), trông hơi 'mịn' hoặc bị bệt vùng má / trán.",
            ),
            (
                "Lighting Error",
                "Ánh sáng trên mặt có lúc lệch nhẹ với cổ hoặc nền, đặc biệt ở các khung hình chuyển động.",
            ),
        ],
    },

    "FaceSwap": {
        "summary": (
            "FaceSwap dùng landmark + blending để thay toàn bộ mặt. "
            "Điểm yếu điển hình là viền mặt, lệch tông màu và méo hình khi đầu quay nhanh."
        ),
        "artifacts": [
            (
                "Boundary Error",
                "Viền mặt rõ, nhất là ở vùng trán – tai – cằm, dễ thấy khi nhân vật xoay hoặc nghiêng đầu.",
            ),
            (
                "Color Mismatch",
                "Màu da khuôn mặt không khớp màu cổ / tai, tạo cảm giác mặt bị 'dán' lên cơ thể.",
            ),
            (
                "Geometry Distortion",
                "Khi quay nhanh, mặt có thể trượt nhẹ so với đầu thật hoặc bị kéo méo.",
            ),
        ],
    },

    "NeuralTextures": {
        "summary": (
            "NeuralTextures học texture 3D của khuôn mặt và chiếu lại. "
            "Điểm yếu chính là texture bị rạn khi quay đầu và ánh sáng không hoàn toàn khớp."
        ),
        "artifacts": [
            (
                "Texture Abnormality",
                "Khi nhân vật xoay đầu, vùng má / trán xuất hiện pattern lạ hoặc rạn texture.",
            ),
            (
                "Color Mismatch",
                "Shade màu trên mặt đôi khi không đúng với phần còn lại của khung hình.",
            ),
            (
                "Lighting Error",
                "Highlight / vùng sáng trên mặt không dịch chuyển đúng theo nguồn sáng khi nhân vật di chuyển.",
            ),
        ],
    },

    "Video2VideoID": {
        "summary": (
            "Video2VideoID tái dựng lại toàn cảnh và nhân dạng giống video gốc nhưng thay đổi nội dung. "
            "Thường lộ lỗi ở nền và hình học khi cảnh / nhân vật di chuyển."
        ),
        "artifacts": [
            (
                "Background Artifact",
                "Nền phía sau nhân vật bị trôi, rung hoặc xuất hiện nhiễu bất thường giữa các khung hình.",
            ),
            (
                "Geometry Distortion",
                "Tỷ lệ cơ thể / đầu thay đổi nhẹ theo thời gian, hoặc hình dạng khuôn mặt méo khi chuyển động.",
            ),
            (
                "Texture Abnormality",
                "Một số vùng cảnh hoặc quần áo bị nhòe, texture không ổn định khi camera/nhân vật di chuyển.",
            ),
        ],
    },
}