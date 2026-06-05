import cv2

def crop_image(image, bbox):

    return image[
        bbox.y1:bbox.y2,
        bbox.x1:bbox.x2
    ]