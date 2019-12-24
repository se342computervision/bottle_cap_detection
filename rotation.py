import numpy as np
from PIL import Image


def get_sat(pix, threshold=100):
    r = pix[:,:,0]
    g = pix[:,:,1]
    b = pix[:,:,2]
    sat = (np.ones_like(r) - np.minimum(np.minimum(r, g), b) / np.maximum(np.maximum(r, g), b)) * 255
    sat = np.where(sat > threshold, 255, 0).astype(np.uint8)
    return sat


def add_margin(pil_img, top=100, right=100, bottom=100, left=100, color=0):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def cut(img, margin=30):
    sat = get_sat(img)
    left = max(int(first_nonzero(np.max(sat, axis=0), 0)) - margin, 0)
    right = min(int(last_nonzero(np.max(sat, axis=0), 0)) + margin, img.shape[1])
    up = max(int(first_nonzero(np.max(sat, axis=1), 0)) - margin, 0)
    down = min(int(last_nonzero(np.max(sat, axis=1), 0)) + margin, img.shape[0])
    return img[up:down, left:right]


def rotate(mat):
    result = []
    for arg in range(0, 360, 15):
        im = Image.fromarray(mat)
        im = add_margin(im)
        im = im.rotate(arg)
        result.append(cut(np.asarray(im), margin=10))
    return result


# if __name__ == "__main__":
#     mat = np.asarray(Image.open("../瓶盖/红色/result/DSC02609-0-4236-3256.jpg"))
#     rotate(mat)