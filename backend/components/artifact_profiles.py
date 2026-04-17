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
            "Audio2Animation là một kỹ thuật dựa trên 3DMM kết hợp với GAN (Generative Adversarial Networks), để sinh biểu cảm khuôn mặt từ giọng nói kết hợp với một hình ảnh. "
            "Thường lộ lỗi ở vùng miệng và chuyển động đầu, vì phải cử động các bộ phận trên khuôn mặt. "
            "Suy ra chuyển động từ âm thanh chứ không dựa vào cảnh thật. "
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
                "Khi lắc đầu hoặc quay nhanh, vùng má / cằm bị kéo giãn hoặc biến dạng nhẹ.",
            ),
        ],
    },

    "Deepfakes": {
        "summary": (
            "Deepfakes cổ điển (autoencoder) thường tạo lớp mặt nạ của đối tượng A dán lên khuôn mặt của đối tượng B. "
            "Điểm yếu rõ nhất là viền mặt, lệch tông màu da và da đôi khi mịn bất thường. "
        ),
        "artifacts": [
            (
                "Boundary Error",
                "Đường viền mặt mờ/nhòe, nhìn có cảm giác như một lớp keo dán lên khung hình.",
            ),
            (
                "Color Mismatch",
                "Tông màu da trên mặt khác với cổ hoặc vùng da xung quanh (sáng hơn, hồng hơn hoặc vàng hơn).",
            ),
            (
                "Texture Abnormality",
                "Da mặt mịn như nhựa, mất chi tiết lỗ chân lông, nếp nhăn so với vùng xung quanh.",
            ),
        ],
    },

    "Face2Face": {
        "summary": (
            "Face2Face điều khiển biểu cảm khuôn mặt A dựa trên biểu cảm gốc của đối tượng B. "
            "Thường lộ lỗi khi miệng hoạt động mạnh và khi ánh sáng / bóng trên mặt không khớp với chuyển động. "
        ),
        "artifacts": [
            (
                "Mouth Artifact",
                "Khi nhân vật cười, nói lớn hoặc phát âm phức tạp, vùng miệng dễ bị méo, môi hoặc răng trông không tự nhiên.",
            ),
            (
                "Lighting Error",
                "Khi nhân vật quay đầu, ánh sáng trên mặt gần như không đổi, không phản xạ theo hướng nguồn sáng của cảnh vật.",
            ),
            (
                "Geometry Distortion",
                "Một số khung hình có méo nhẹ vùng cằm/ má khi biểu cảm thay đổi nhanh.",
            ),
        ],
    },

    "FaceShifter": {
        "summary": (
            "FaceShifter là GAN (Generative Adversarial Networks) chất lượng cao, che giấu lỗi tốt hơn nhưng vẫn rất yếu ở vùng mắt. "
            "Kết cấu da không tự nhiên, một số vùng da tràn ra khỏi viền khuôn mặt khi quan sát kỹ. "
        ),
        "artifacts": [
            (
                "Eye Artifact",
                "Độ sắc nét/độ sáng hai mắt không đồng đều, phản chiếu trong mắt không khớp với ánh sáng chung.",
            ),
            (
                "Texture Abnormality",
                "Da thiếu các chi tiết rất nhỏ, trông hơi 'mịn' hoặc bị bệt vùng má / trán, và tràn ra khỏi viền khuôn mặt.",
            ),
            (
                "Lighting Error",
                "Ánh sáng trên mặt có lúc lệch nhẹ với cổ hoặc nền, đặc biệt ở các khung hình chuyển động phức tạp.",
            ),
        ],
    },

    "FaceSwap": {
        "summary": (
            "FaceSwap là một kỹ thuật dùng các mốc và trộn, để thay toàn bộ khuôn mặt của đối tượng A sang cho phần đầu của đối tượng B. "
            "Điểm yếu điển hình là viền mặt, lệch tông màu và méo hình khi đầu quay nhanh. "
        ),
        "artifacts": [
            (
                "Boundary Error",
                "Viền mặt rõ, nhất là ở vùng trán, tai, cằm, dễ thấy khi nhân vật xoay hoặc nghiêng đầu.",
            ),
            (
                "Color Mismatch",
                "Màu da khuôn mặt không khớp màu cổ / tai, tạo cảm giác khuôn mặt bị 'dán' lên cả cái đầu.",
            ),
            (
                "Geometry Distortion",
                "Khi quay nhanh, mặt có thể trượt nhẹ so với phần đầu thật hoặc bị kéo méo.",
            ),
        ],
    },

    "NeuralTextures": {
        "summary": (
            "NeuralTextures là một kỹ thuật dựa trên khả năng tạo ra kết cấu 3D của khuôn mặt người thật. Đây là loại rất khó để nhận biết bằng mắt thường. "
            "Điểm yếu chính là kết cấu bị rạn khi quay đầu và ánh sáng không hoàn toàn khớp. "
        ),
        "artifacts": [
            (
                "Texture Abnormality",
                "Khi nhân vật xoay đầu, vùng má / trán xuất hiện làn da lạ như bị rạn nứt.",
            ),
            (
                "Color Mismatch",
                "Tông màu trên mặt đôi khi không đúng với phần còn lại của phần đầu.",
            ),
            (
                "Lighting Error",
                "Các vùng sáng trên mặt không dịch chuyển đúng theo nguồn sáng xung quanh khi nhân vật chuyển động.",
            ),
        ],
    },

    "Video2VideoID": {
        "summary": (
            "Video2VideoID là một kỹ thuật dựa trên mô hình Diffusion, sẽ nhận vào một nguồn ảnh hoặc video chứa khuôn mặt của một người, và tạo ra video mới với chính khuôn mặt đó nhưng nội dung, biểu cảm, hành động hoàn toàn khác — người đó có thể đang nói, di chuyển hoặc xuất hiện trong bối cảnh hoàn toàn bịa đặt không hề có thật trong đời thực. "
            "Thường lộ lỗi ở các cảnh nền khi cảnh vật và nhân vật di chuyển. "
        ),
        "artifacts": [
            (
                "Background Artifact",
                "Nền phía sau nhân vật bị trôi, rung hoặc xuất hiện nhiễu và mờ nhòe bất thường giữa các khung hình.",
            ),
            (
                "Geometry Distortion",
                "Tỷ lệ cơ thể / đầu thay đổi nhẹ theo thời gian, hoặc hình dạng khuôn mặt méo khi chuyển động.",
            ),
            (
                "Texture Abnormality",
                "Một số vùng cảnh hoặc quần áo bị nhòe, kết cấu hình ảnh không ổn định khi camera/nhân vật di chuyển.",
            ),
        ],
    },
}