from PIL import Image
import numpy as np
import cv2
import copy


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(
    image,
    face_box,
    upper_boundary_ratio=0.5,
    expand=1.5,
    fp=None,
    mode="raw",
    mask_blur_ratio=0.1,
    lips_dilation=0,
    aux_modes=("lips_only", "skin_only"),
):
    """Build the blend mask and expanded crop box for paste-back.

    For ``mode="lips_only"`` and ``mode="lips_outer_only"`` the
    mask is restricted to the lips, bypasses the upper-boundary
    crop (lips can be at any vertical position), and optionally
    dilated by a small elliptical kernel
    (``lips_dilation > 0``) so the blend edge sits a few pixels
    outside the actual lip boundary. The same ``mask_blur_ratio``
    Gaussian pass still runs.

    ``lips_outer_only`` uses parser classes 12 + 13 (vermilion
    only) and excludes the mouth interior (11), so the source's
    open-mouth dark area is not overwritten by the model's wider
    dark interior.

    The function runs BiSeNet **at most once** for the face crop
    on this frame. The legacy implementation called
    ``face_seg(face_large, mode=mode)`` here and then the
    caller re-invoked the parser two more times on the same
    crop for ``skin_only`` and ``lips_only`` -- 3 forwards per
    source frame. Now we ask the parser for the raw class map
    once via ``parse_classes`` and derive the primary mask plus
    any ``aux_modes`` (e.g. ``lips_only``, ``skin_only``) from
    it via ``_build_mode_mask``. The returned ``aux`` dict maps
    mode name -> uint8 mask of the same shape as the primary
    mask; pass an empty tuple to skip the extra numpy work
    entirely.
    """
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    # Single BiSeNet forward. From this class map we build the
    # primary mask (post-blend-mask pipeline) AND any auxiliary
    # masks the caller requested -- a typical request asks for
    # ("lips_only", "skin_only") so the post-process pipeline can
    # do color match / detail restore / badcase gates without
    # re-running the network.
    if fp is None or not hasattr(fp, "parse_classes"):
        # Fallback: legacy path that re-runs the parser per call.
        # Kept so unit tests / direct callers that pass a duck-
        # typed ``fp`` still work.
        mask_image = face_seg(face_large, mode=mode, fp=fp)
        aux: dict = {}
    else:
        parsing = fp.parse_classes(face_large)
        aux = {
            aux_mode: fp._build_mode_mask(parsing, aux_mode)
            for aux_mode in aux_modes
        }
        primary = fp._build_mode_mask(parsing, mode)
        mask_image = Image.fromarray(primary)

    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    if mode in ("lips_only", "lips_outer_only"):
        # Lips can be at any vertical position, so we skip the
        # upper-boundary crop that "jaw" / "raw" modes use. The
        # caller controls dilation via lips_dilation; 0 keeps the
        # mask exactly at the parser's lip boundary (tightest
        # result), 2-3 adds a small safety margin for generated
        # lip texture right at the boundary.
        mask_array = np.array(mask_image)
        if lips_dilation > 0:
            k = 2 * lips_dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        if mask_blur_ratio > 0:
            blur_kernel_size = int(mask_blur_ratio * ori_shape[0] // 2 * 2) + 1
            mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)
        return mask_array, crop_box, aux

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    mask_array = np.array(modified_mask_image)
    if mask_blur_ratio > 0:
        blur_kernel_size = int(mask_blur_ratio * ori_shape[0] // 2 * 2) + 1
        mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box, aux
