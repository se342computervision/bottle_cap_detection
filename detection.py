import selectivesearch

from sift import *


def confidence(img, x, y, w, h):
    if w * h > 200000:
        return 0
    else:
        return w * h


def get_sat(pix, threshold=100):
    r = pix[:, :, 0]
    g = pix[:, :, 1]
    b = pix[:, :, 2]
    sat = (np.ones_like(r) - np.minimum(np.minimum(r, g), b) / np.maximum(np.maximum(r, g), b)) * 255
    sat = np.where(sat > threshold, 255, 0).astype(np.uint8)
    return sat


def search(filename):
    # loading image
    im = Image.open(filename)
    (x, y) = im.size  # 读取图片大小
    new_x = x // 4
    new_y = y // 4
    raw_img = np.asarray(im)
    im = im.resize((new_x, new_y), Image.ANTIALIAS)
    resized_img = np.asarray(im)
    sat = get_sat(resized_img)
    img = np.stack((sat,) * 3, axis=-1)
    Image.fromarray(img).save("1.jpg")

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = list()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions too small or too large
        if r['size'] < 2000 or r['size'] > 0.5 * img.shape[0] * img.shape[1]:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 4 or h / w > 4:
            continue
        candidates.append((x * 4, y * 4, w * 4, h * 4))

    return raw_img, candidates


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2] + boxes[:, 0]
    end_y = boxes[:, 3] + boxes[:, 1]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def is_updown(img):
    sat = get_sat(img)
    val = []
    for i in range(180):
        im = Image.fromarray(sat)
        im = add_margin(im, 100, 100, 100, 100, 0)
        pix = np.asarray(im.rotate(i))
        val.append(np.sum(pix[pix.shape[0] // 2, :]))

    return np.std(np.asarray(val)) < 10000


def rotate(img):
    sat = get_sat(img)
    min_pos = 0
    min_dev = 1000000000
    for i in range(180):
        im = Image.fromarray(sat)
        im = add_margin(im, 100, 100, 100, 100, 0)
        pix = np.asarray(im.rotate(i))
        dev = np.std(np.trim_zeros(first_nonzero(pix, 0, invalid_val=0))) + \
              np.std(np.trim_zeros(last_nonzero(pix, 0, invalid_val=0)))
        if dev < min_dev:
            min_dev = dev
            min_pos = i

    rotated_pix1 = np.asarray(add_margin(Image.fromarray(sat), 100, 100, 100, 100, 0).rotate(min_pos))
    rotated_pix1 = np.sum(rotated_pix1, axis=0)
    rotated_pix1 = np.trim_zeros(rotated_pix1)
    score1 = np.average(rotated_pix1)

    rotated_pix2 = np.asarray(add_margin(Image.fromarray(sat), 100, 100, 100, 100, 0).rotate(min_pos).rotate(90))
    rotated_pix2 = np.sum(rotated_pix2, axis=0)
    rotated_pix2 = np.trim_zeros(rotated_pix2)
    score2 = np.average(rotated_pix2)

    rotated_pix1 = np.asarray(add_margin(Image.fromarray(sat), 100, 100, 100, 100, 0).rotate(min_pos))
    rotated_pix2 = np.asarray(add_margin(Image.fromarray(sat), 100, 100, 100, 100, 0).rotate(min_pos).rotate(90))
    score3 = np.max(np.trim_zeros(last_nonzero(rotated_pix1, 0, invalid_val=0))) - np.min(
        np.trim_zeros(first_nonzero(rotated_pix1, 0, invalid_val=10000)))
    score4 = np.max(np.trim_zeros(last_nonzero(rotated_pix2, 0, invalid_val=0))) - np.min(
        np.trim_zeros(first_nonzero(rotated_pix2, 0, invalid_val=10000)))

    if score4 / score3 < 1.3 and score3 / score4 < 1.3:
        return 0, True
    elif score1 < score2:
        return min_pos, False
    else:
        return (min_pos + 90) % 180, False


def detection(file, margin=10):
    img, candidates = search(file)

    confidences = [confidence(img, each[0], each[1], each[2], each[3]) for each in candidates]
    picked, _ = nms(candidates, confidences, 0.1)
    cap = list()
    for x, y, w, h in picked:
        cap.append((img[y - margin:y + h + margin, x - margin:x + w + margin], (x, y, w, h)))
    return cap


def cut(img, margin=30):
    sat = get_sat(img)
    left = max(int(first_nonzero(np.max(sat, axis=0), 0)) - margin, 0)
    right = min(int(last_nonzero(np.max(sat, axis=0), 0)) + margin, img.shape[1])
    up = max(int(first_nonzero(np.max(sat, axis=1), 0)) - margin, 0)
    down = min(int(last_nonzero(np.max(sat, axis=1), 0)) + margin, img.shape[0])
    return img[up:down, left:right]


def coloring(filename, match_info):
    raw = Image.open(filename)
    raw_mat = np.asarray(raw).copy()
    for pos, degree, _, mat, mask, origin_point in match_info:
        if degree > 90:
            degree -= 180
            mask = np.rot90(np.rot90(mask))

        l = pos[0]
        h = pos[1]
        rad = np.deg2rad(degree)
        mask = np.asarray(Image.fromarray(mask).resize((mat.shape[1], mat.shape[0])))
        x0s, y0s = np.where(mask != 0)
        xs = (x0s * np.cos(rad) + y0s * np.sin(rad) + h)
        ys = (-x0s * np.sin(rad) + y0s * np.cos(rad) + l)
        xs = xs.astype(np.uint32)
        ys = ys.astype(np.uint32)
        raw_mat[xs, ys] = np.asarray([0, 0, 0])
    return Image.fromarray(raw_mat)


def run(filename):
    query_img0, query_img_hog0, query_img_name0, kp0, des0 = sift_init()
    result = []
    for mat, pos in detection(filename):
        im1 = Image.fromarray(mat)
        im2 = add_margin(im1, 100, 100, 100, 100, 0)
        degree, updown = rotate(mat)
        if not updown:
            mat = cut(np.asarray(im2.rotate(degree)))
        selected0, img_mask0, origin_point0 = sift_match(mat, query_img0, query_img_hog0, query_img_name0,
                                                         kp0, des0)
        result.append((pos, degree, selected0, mat, img_mask0, origin_point0))
    return coloring(filename, result)


if __name__ == "__main__":
    run("your_filename_here.jpg")