# proposal/proposal_generator.py

from crop.candidate_generator import (
    generate_candidates
)

from proposal.saliency_center_guided import (
    generate_saliency_center_boxes
)

from proposal.semantic_center_guided import (
    generate_semantic_center_boxes
)


def generate_all_proposals(
    img_rgb,
    saliency_map,
    segments
):
    """
    返回所有候选框
    """

    h, w = img_rgb.shape[:2]

    # ------------------
    # Grid
    # ------------------

    grid_boxes = generate_candidates(
        img_w=w,
        img_h=h
    )

    # ------------------
    # Saliency
    # ------------------

    saliency_boxes = (
        generate_saliency_center_boxes(
            saliency_map,
            w,
            h
        )
    )

    # ------------------
    # Semantic
    # ------------------

    semantic_boxes = (
        generate_semantic_center_boxes(
            segments,
            w,
            h
        )
    )

    all_boxes = (
        grid_boxes
        + saliency_boxes
        + semantic_boxes
    )

    print()

    print("Grid      :", len(grid_boxes))
    print("Saliency  :", len(saliency_boxes))
    print("Semantic  :", len(semantic_boxes))
    print("Total     :", len(all_boxes))

    return all_boxes