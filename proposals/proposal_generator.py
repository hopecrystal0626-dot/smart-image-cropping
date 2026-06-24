# proposal/proposal_generator.py
from smart_framing.crop.candidate_generator import generate_candidates
from smart_framing.proposals.saliency_center_guided import generate_saliency_center_boxes
from smart_framing.proposals.semantic_center_guided import generate_semantic_center_boxes

def generate_all_proposals(img_rgb, saliency_map, segments):
    h, w = img_rgb.shape[:2]
    grid_boxes = generate_candidates(img_w=w, img_h=h)
    saliency_boxes = generate_saliency_center_boxes(saliency_map, w, h)
    semantic_boxes = generate_semantic_center_boxes(segments, w, h)
    all_boxes = grid_boxes + saliency_boxes + semantic_boxes
    print(f"Grid: {len(grid_boxes)}, Saliency: {len(saliency_boxes)}, Semantic: {len(semantic_boxes)}, Total: {len(all_boxes)}")
    return all_boxes