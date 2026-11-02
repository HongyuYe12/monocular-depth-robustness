from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .subject import SubjectCandidate


@dataclass(frozen=True)
class DepthProfile:
    """一个主体框内部及其周围背景的相对深度统计。

    median: 主体区域深度中位数，用于降低少量错误像素的影响。
    spread: 主体内部四分位深度跨度，占全图有效深度范围的比例。
    separation: 主体与周围背景的深度分离程度，越大越容易独立对焦。
    consistency: 主体框内部是否接近单一深度层，越大越一致。
    multi_depth_ambiguity: 框内可能同时覆盖前景、目标和背景。
    """

    median: float
    lower_quartile: float
    upper_quartile: float
    spread: float
    normalized_median: float
    background_median: float
    separation: float
    consistency: float
    multi_depth_ambiguity: bool


@dataclass(frozen=True)
class RankedSubject:
    """完成深度分析和自动排序后的主体。"""

    subject: SubjectCandidate
    depth: DepthProfile
    center_score: float
    area_score: float
    class_preference: float
    total_score: float
    focus_zone: str
    focus_confidence: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["subject"]["box"] = list(self.subject.box)
        return result


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """把检测框限制在深度图范围内。"""
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _ring_mask(shape: tuple[int, int], box: tuple[int, int, int, int], padding: float = 0.18) -> np.ndarray:
    """建立主体框外围的一圈背景掩码，用于比较主体与附近背景深度。"""
    height, width = shape
    x1, y1, x2, y2 = _clip_box(box, width, height)
    pad_x = max(2, round((x2 - x1) * padding))
    pad_y = max(2, round((y2 - y1) * padding))
    outer = np.zeros(shape, dtype=bool)
    # 先填充扩大的外框，再扣除主体框本身，得到环形背景区域。
    outer[max(0, y1 - pad_y):min(height, y2 + pad_y), max(0, x1 - pad_x):min(width, x2 + pad_x)] = True
    outer[y1:y2, x1:x2] = False
    return outer


def analyze_subject_depth(
    depth: np.ndarray,
    subject: SubjectCandidate,
    *,
    ambiguity_threshold: float = 0.22,
) -> DepthProfile:
    """计算单个主体的深度分布、背景分离度和多深度歧义。"""
    if depth.ndim != 2:
        raise ValueError("depth map must be two-dimensional")
    height, width = depth.shape
    x1, y1, x2, y2 = _clip_box(subject.box, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty subject box: {subject.box}")

    region = depth[y1:y2, x1:x2].astype(np.float64)
    region = region[np.isfinite(region)]
    if region.size < 4:
        raise ValueError("subject region has too few finite depth pixels")

    global_values = depth[np.isfinite(depth)].astype(np.float64)
    # 使用 2%–98% 分位数作为全图有效范围，减少极端错误深度值的影响。
    global_low, global_high = np.percentile(global_values, [2, 98])
    global_range = max(float(global_high - global_low), 1e-12)
    # 四分位距比最大值减最小值更稳健，适合检测主体框是否跨越多个深度层。
    q25, median, q75 = np.percentile(region, [25, 50, 75])
    spread = float((q75 - q25) / global_range)

    # 比较主体中位深度与附近一圈背景，而不是与整个画面背景比较。
    ring_values = depth[_ring_mask(depth.shape, subject.box)]
    ring_values = ring_values[np.isfinite(ring_values)]
    background_median = float(np.median(ring_values)) if ring_values.size else float(np.median(global_values))
    separation = min(1.0, abs(float(median) - background_median) / global_range)
    normalized_median = float(np.clip((median - global_low) / global_range, 0.0, 1.0))
    # spread 达到阈值时 consistency 下降到 0，并触发多深度歧义标记。
    consistency = float(np.clip(1.0 - spread / max(ambiguity_threshold, 1e-12), 0.0, 1.0))

    return DepthProfile(
        median=float(median),
        lower_quartile=float(q25),
        upper_quartile=float(q75),
        spread=spread,
        normalized_median=normalized_median,
        background_median=background_median,
        separation=separation,
        consistency=consistency,
        multi_depth_ambiguity=spread >= ambiguity_threshold,
    )


def focus_zone(normalized_depth: float, *, larger_is_near: bool = True) -> str:
    """把相对深度粗分为近景、中景和远景，作为镜头搜索初始区域。

    larger_is_near 必须按具体深度模型确认。若模型数值越大表示越远，应设为 False。
    这里输出的不是米制距离，也不是镜头马达位置。
    """
    nearness = normalized_depth if larger_is_near else 1.0 - normalized_depth
    if nearness >= 2 / 3:
        return "near"
    if nearness >= 1 / 3:
        return "mid"
    return "far"


def rank_subjects(
    depth: np.ndarray,
    subjects: list[SubjectCandidate],
    *,
    preferences: dict[str, float],
    weights: dict[str, float],
    ambiguity_threshold: float = 0.22,
    larger_depth_is_near: bool = True,
) -> list[RankedSubject]:
    """综合检测、用户偏好、构图和深度信息，对候选主体自动排序。"""
    height, width = depth.shape
    image_area = max(1, width * height)
    ranked: list[RankedSubject] = []

    for subject in subjects:
        profile = analyze_subject_depth(depth, subject, ambiguity_threshold=ambiguity_threshold)
        x1, y1, x2, y2 = subject.box
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        # 越靠近画面中心，center_score 越高；它只是一项弱先验。
        distance_from_center = min(1.0, np.hypot(center_x - 0.5, center_y - 0.5) / np.hypot(0.5, 0.5))
        center_score = float(1.0 - distance_from_center)
        # 对面积比例开平方，避免大主体的面积优势压倒其他评分项。
        area_score = float(np.clip(((x2 - x1) * (y2 - y1) / image_area) ** 0.5, 0.0, 1.0))
        class_preference = float(preferences.get(subject.label, preferences.get("default", 0.35)))

        # 所有分量尽量保持在 0–1 范围，权重可在 configs/autofocus.json 修改。
        components = {
            "detection": subject.confidence,
            "class_preference": class_preference,
            "center": center_score,
            "area": area_score,
            "depth_separation": profile.separation,
            "depth_consistency": profile.consistency,
        }
        weight_total = sum(float(weights.get(key, 0.0)) for key in components)
        if weight_total <= 0:
            raise ValueError("score weights must contain a positive value")
        total_score = sum(float(weights.get(key, 0.0)) * value for key, value in components.items()) / weight_total
        # 焦距建议置信度更看重主体内部是否属于单一深度层。
        # 若置信度低，主流程会要求拍摄验证帧，而不直接信任深度建议。
        confidence = float(np.clip(0.55 * profile.consistency + 0.30 * profile.separation + 0.15 * subject.confidence, 0, 1))

        ranked.append(
            RankedSubject(
                subject=subject,
                depth=profile,
                center_score=center_score,
                area_score=area_score,
                class_preference=class_preference,
                total_score=float(total_score),
                focus_zone=focus_zone(profile.normalized_median, larger_is_near=larger_depth_is_near),
                focus_confidence=confidence,
            )
        )

    return sorted(ranked, key=lambda item: item.total_score, reverse=True)
