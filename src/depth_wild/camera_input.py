from __future__ import annotations

from pathlib import Path

from PIL import Image

from .io import load_rgb

# 当前先覆盖常见相机 RAW 后缀。后续接入相机 SDK 时，可以绕过文件后缀，
# 直接把相机缓存中的预览帧或 RAW 缓冲区转换为 PIL.Image。
RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}


def load_camera_frame(path: Path) -> Image.Image:
    """读取普通图片，或将 RAW 显影为模型可以使用的 RGB 图像。

    RAW 在本项目中是低照度输入增强方式，不是主体选择和深度引导的必要条件。
    这里不会修改原始传感器文件。未来连接相机 SDK 时，可以将本函数替换为
    从内存预览帧读取的实现。
    """
    # JPG/PNG 等普通图片沿用原项目的 EXIF 方向修正与 RGB 转换。
    if path.suffix.lower() not in RAW_SUFFIXES:
        return load_rgb(path)

    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError("RAW input requires: pip install -e .[raw]") from error

    with rawpy.imread(str(path)) as raw:
        # 使用相机记录的白平衡，将 RAW 转成 8 位 RGB 供现有预训练模型推理。
        # 后续研究低照度时，可以在这里替换为专门的 RAW 降噪/多帧融合流程。
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
        )
    return Image.fromarray(rgb).convert("RGB")
