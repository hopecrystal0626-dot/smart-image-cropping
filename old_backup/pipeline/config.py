from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CandidateConfig:
    scale_grid: Dict[float, int] = field(default_factory=lambda: {
        0.20: 12,
        0.25: 12,
        0.30: 12,
        0.35: 11,
        0.40: 11,
        0.50: 8,
    })
    jitter_ratio: float = 0.05
    seed: int = 42


@dataclass
class SaliencyConfig:
    top_percent: float = 0.30
    num_segments: int = 3


@dataclass
class CompletenessConfig:
    yolo_conf_threshold: float = 0.30
    completeness_threshold: float = 0.85
    min_area_ratio: float = 0.10
    human_coverage_threshold: float = 0.90
    object_coverage_threshold: float = 0.85
    max_edge_touch_ratio: float = 0.25
    edge_margin_px: int = 12
    structure_gate_threshold: float = 0.55


@dataclass
class SceneConfig:
    vertical_edge_ratio_threshold: float = 0.62
    horizontal_edge_ratio_threshold: float = 0.55
    building_structure_threshold: float = 0.50
    structure_candidate_max: int = 6
    structure_pad_ratio: float = 0.12


@dataclass
class FusionConfig:
    alpha: float = 0.40
    beta: float = 0.60
    top_k: int = 10
    clip_mode: str = "balanced"


@dataclass
class StepWeights:
    w_human: float = 1.0
    w_object: float = 0.8
    w_center: float = 0.2
    w_structure: float = 0.35
    min_cover_thresh: float = 0.3
    expand_ratio: float = 0.15
    target_area_ratio: float = 0.30
    area_low: float = 0.20
    area_high: float = 0.50


@dataclass
class PipelineConfig:
    yolo_model_path: str = "yolov8n.pt"
    output_dir: str = "./data/output/systemized"
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    weights: StepWeights = field(default_factory=StepWeights)

    def to_dict(self) -> Dict:
        return {
            "yolo_model_path": self.yolo_model_path,
            "output_dir": self.output_dir,
            "candidate": self.candidate.__dict__,
            "saliency": self.saliency.__dict__,
            "completeness": self.completeness.__dict__,
            "scene": self.scene.__dict__,
            "fusion": self.fusion.__dict__,
            "weights": self.weights.__dict__,
        }


DEFAULT_PIPELINE_CONFIG = PipelineConfig()
