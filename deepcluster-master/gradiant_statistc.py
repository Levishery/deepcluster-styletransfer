import cv2
import numpy as np

def calc_phase(img):
    max = np.max(img)
    img = np.sqrt(img / float(np.max(img)))
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)

    bin_size = 16
    angle_unit = 360 / bin_size
    orientation_centers = [0] * bin_size

    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            gradient_strength = gradient_magnitude[k][l]
            g_angle = gradient_angle[k][l]
            min_angle = int(g_angle / angle_unit) % 16
            max_angle = (min_angle + 1) % bin_size
            mod = g_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    center_angle = angle_unit * np.argmax(orientation_centers)
    return center_angle


crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]


def rotate_image(img, angle, crop):
    """rotate a img and crop out black padding, then resize the img to the original size."""

    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

    if crop:
        angle_crop = angle % 180

        if angle_crop > 90:
            angle_crop = 180 - angle_crop

        theta = angle_crop * np.pi / 180.0

        hw_ratio = float(h) / float(w)

        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta

        r = hw_ratio if h > w else 1 / hw_ratio

        denominator = r * tan_theta + 1

        crop_mult = numerator / denominator

        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
        img_rotated = cv2.resize(img_rotated, img.shape[:2])

    return img_rotated

