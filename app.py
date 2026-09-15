

import os
import time
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ── CONFIG ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_DIR = os.path.join(BASE_DIR, "model")

# Ưu tiên model/model.pt theo cấu trúc project gốc, fallback sang model.pt cạnh app.py.
YOLO_MODEL_CANDIDATES = [
    os.path.join(CNN_MODEL_DIR, "model.pt"),
    os.path.join(BASE_DIR, "model.pt"),
]
YOLO_MODEL_PATH = next((p for p in YOLO_MODEL_CANDIDATES if os.path.exists(p)), YOLO_MODEL_CANDIDATES[0])

IMG_HEIGHT, IMG_WIDTH = 128, 128
STD_W, STD_H = 800, 640

YOLO_CONF = 0.45
YOLO_IOU = 0.45
EGG_CLASS_ID = 0

# Giữ nguyên thứ tự neuron output của global2.h5. KHÔNG đổi thứ tự này.
GLOBAL_CLASSES = ["Canh", "Chien", "Com", "Khac", "Kho", "Nuong", "Xao"]

# Class Khac trong model thực chất là món Đậu hũ sốt cà.
CLASS_LABEL_ALIASES = {
    "Khac": "Đậu hũ sốt cà",
    "Đậu hũ sốt cà chua": "Đậu hũ sốt cà",
    "Đậu hũ sốt cà": "Đậu hũ sốt cà",
}

SUB_MODELS = {
    "Canh": {
        "model": "Canh_CNN_Best(v2).h5",
        "classes": [
            "Canh chua có cá",
            "Canh chua không cá",
            "Canh rau cải thảo",
            "Canh rau muống",
        ],
    },
    "Chien": {
        "model": "Chien_CNN_Best.h5",
        "classes": ["Trứng chiên", "Trứng chiên thịt"],
    },
    "Kho": {
        "model": "Kho_CNN_Best(v2).h5",
        "classes": ["Cá hú kho", "Thịt kho"],
    },
    "Xao": {
        "model": "Xao_CNN_Best(v2).h5",
        "classes": [
            "Rau xào củ sắn",
            "Rau xào đậu dừa",
            "Rau xào đậu que",
            "Rau xào Lagim",
        ],
    },
}

# Fallback crop coordinates on standardized 800x640 image.
# Frontend mới gửi interactive boxes nên phần này chỉ để tương thích bản cũ.
COORDS_TOP3 = [
    ("Món 1", [(55, 65), (240, 65), (240, 255), (55, 255)]),
    ("Món 2", [(300, 65), (500, 65), (500, 255), (300, 255)]),
    ("Món 3", [(565, 65), (745, 65), (745, 255), (565, 255)]),
    ("Món 4", [(60, 310), (330, 310), (330, 575), (60, 575)]),
    ("Món 5", [(500, 310), (745, 310), (745, 575), (500, 575)]),
]


def rotate_pts_180(pts):
    rot = [(STD_W - x, STD_H - y) for x, y in pts]
    return [rot[2], rot[3], rot[0], rot[1]]


COORDS_PRESETS = {
    "top3": COORDS_TOP3,
    "bottom3": [(name, rotate_pts_180(pts)) for name, pts in COORDS_TOP3],
}

# ── BẢNG GIÁ ──────────────────────────────────────────────────────────
PRICE_TABLE = {
    "Com": 10_000,
    "Cá hú kho": 30_000,
    "Thịt kho": 25_000,              # không có trứng
    "Thịt kho trứng": 30_000,        # base gồm 1 trứng
    "Canh chua có cá": 25_000,
    "Canh chua không cá": 10_000,
    "Canh rau cải thảo": 7_000,
    "Canh rau muống": 7_000,
    "Nuong": 30_000,
    "Rau xào củ sắn": 10_000,
    "Rau xào đậu dừa": 10_000,
    "Rau xào đậu que": 10_000,
    "Rau xào Lagim": 10_000,
    "Trứng chiên": 25_000,
    "Trứng chiên thịt": 25_000,
    "Khac": 25_000,
    "Đậu hũ sốt cà": 25_000,
    "Đậu hũ sốt cà chua": 25_000,
}

THIT_KHO_NO_EGG_PRICE = 25_000
THIT_KHO_EGG_BASE_PRICE = 30_000
EGG_BASE_COUNT = 1
EGG_EXTRA_PRICE = 6_000

# ── LOAD MODELS ───────────────────────────────────────────────────────

class CompatBatchNorm(BatchNormalization):
    def __init__(self, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(**kwargs)


custom_obj = {"BatchNormalization": CompatBatchNorm}
tf.get_logger().setLevel("ERROR")

print("=" * 60)
print("Loading Global CNN Model...")
global_model = load_model(
    os.path.join(CNN_MODEL_DIR, "global2.h5"),
    custom_objects=custom_obj,
    compile=False,
)
print("✓ Global CNN loaded")

print("Loading Sub CNN Models...")
loaded_sub_models = {}
for group, cfg in SUB_MODELS.items():
    path = os.path.join(CNN_MODEL_DIR, cfg["model"])
    loaded_sub_models[group] = load_model(path, custom_objects=custom_obj, compile=False)
    print(f"  ✓ {group}")

print("Loading YOLO Egg Detection Model...")
yolo_model = None
yolo_status = "disabled"
yolo_error = ""
if YOLO is None:
    yolo_status = "missing_ultralytics"
    yolo_error = "Chưa cài ultralytics. Chạy: pip install ultralytics"
    print("⚠ YOLO disabled:", yolo_error)
elif not os.path.exists(YOLO_MODEL_PATH):
    yolo_status = "missing_model"
    yolo_error = f"Không tìm thấy YOLO model tại {YOLO_MODEL_PATH}"
    print("⚠ YOLO disabled:", yolo_error)
else:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_status = "ready"
    print(f"✓ YOLO Egg model loaded: {YOLO_MODEL_PATH}")

print("All models initialized.\n")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────

def decode_base64_image(data_url):
    img_b64 = data_url.split(",")[-1]
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def resize_and_crop_center(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def crop_item_perspective(img, four_pts):
    pts = np.float32(four_pts)
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))


def center_crop_square(img):
    """Lấy hình vuông lớn nhất ở chính giữa ảnh, không padding và không kéo méo."""
    if img is None or img.size == 0:
        raise ValueError("Ảnh crop không hợp lệ")

    h, w = img.shape[:2]
    side = min(h, w)
    if side < 12:
        raise ValueError(f"Vùng crop quá nhỏ: {w}x{h}")

    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return img[y1:y1 + side, x1:x1 + side].copy()


def crop_item_rect(img, rect):
    """
    Cắt vùng tương tác theo hình vuông 1:1.

    Nếu frontend hoặc dữ liệu localStorage cũ gửi khung chữ nhật, backend lấy
    hình vuông lớn nhất nằm chính giữa khung đó. Phần dư hai bên/trên dưới bị bỏ,
    tuyệt đối không thêm padding và không kéo giãn ảnh.
    """
    img_h, img_w = img.shape[:2]
    x = int(round(float(rect.get("x", 0))))
    y = int(round(float(rect.get("y", 0))))
    w = int(round(float(rect.get("w", 0))))
    h = int(round(float(rect.get("h", 0))))

    x1 = max(0, min(img_w - 1, x))
    y1 = max(0, min(img_h - 1, y))
    x2 = max(0, min(img_w, x + w))
    y2 = max(0, min(img_h, y + h))

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    side = min(clipped_w, clipped_h)
    if side < 12:
        raise ValueError(f"Vùng crop quá nhỏ hoặc nằm ngoài ảnh: {rect}")

    square_x1 = x1 + (clipped_w - side) // 2
    square_y1 = y1 + (clipped_h - side) // 2
    square_x2 = square_x1 + side
    square_y2 = square_y1 + side

    square_crop = img[square_y1:square_y2, square_x1:square_x2].copy()
    square_rect = {
        "x": int(square_x1),
        "y": int(square_y1),
        "w": int(side),
        "h": int(side),
    }
    return square_crop, square_rect


def preprocess_food_lighting(crop_bgr):
    """
    Giảm ảnh hưởng của đèn chiếu/cháy sáng trước khi đưa ảnh vào CNN.

    Nguyên tắc:
      - Ảnh bình thường được giữ gần như nguyên trạng.
      - Chỉ hiệu chỉnh kênh độ sáng L trong không gian LAB để hạn chế lệch màu.
      - Gamma thích ứng nén vùng sáng; CLAHE nhẹ phục hồi tương phản cục bộ.
      - Không cố "vẽ lại" vùng đã cháy trắng hoàn toàn vì dữ liệu đã mất.

    Trả về:
      processed_bgr: ảnh dùng riêng cho CNN
      info: thông tin nội bộ để kiểm tra chất lượng ánh sáng
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Ảnh crop không hợp lệ")

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    mean_v = float(np.mean(value))
    median_v = float(np.median(value))
    p90_v = float(np.percentile(value, 90))
    p98_v = float(np.percentile(value, 98))

    # Pixel gần trắng và ít bão hòa thường là vùng phản sáng từ đèn.
    glare_mask = (value >= 238.0) & (saturation <= 80.0)
    severe_glare_mask = (value >= 250.0) & (saturation <= 55.0)
    glare_ratio = float(np.mean(glare_mask))
    severe_glare_ratio = float(np.mean(severe_glare_mask))

    needs_correction = (
        median_v >= 185.0
        or mean_v >= 190.0
        or p90_v >= 235.0
        or glare_ratio >= 0.012
    )

    if not needs_correction:
        return crop_bgr.copy(), {
            "corrected": False,
            "quality": "normal",
            "mean_brightness": round(mean_v, 2),
            "median_brightness": round(median_v, 2),
            "glare_ratio": round(glare_ratio, 4),
            "severe_glare_ratio": round(severe_glare_ratio, 4),
            "gamma": 1.0,
        }

    # Gamma > 1 làm tối vùng sáng. Mức gamma tăng theo độ sáng và tỷ lệ glare.
    brightness_pressure = max(0.0, median_v - 145.0) / 95.0
    highlight_pressure = max(0.0, p90_v - 220.0) / 35.0
    glare_pressure = min(1.0, glare_ratio / 0.12)
    gamma = 1.0 + 0.24 * brightness_pressure + 0.22 * highlight_pressure + 0.18 * glare_pressure
    gamma = float(np.clip(gamma, 1.05, 1.58))

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    # Nén thêm điểm trắng để pixel quá sáng không còn chiếm ưu thế tuyệt đối.
    # Đây không khôi phục chi tiết đã cháy trắng, nhưng giúp CNN bớt bị vùng trắng lấn át.
    white_scale = 1.0 - 0.055 * highlight_pressure - 0.045 * glare_pressure
    white_scale = float(np.clip(white_scale, 0.90, 1.0))

    gamma_lut = np.array([
        np.clip(((i / 255.0) ** gamma) * 255.0 * white_scale, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    lightness_gamma = cv2.LUT(lightness, gamma_lut)

    # CLAHE nhẹ giúp lấy lại texture sau khi nén vùng sáng mà không làm ảnh gắt.
    clahe = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8))
    lightness_clahe = clahe.apply(lightness_gamma)

    # Ảnh càng chói thì dùng CLAHE nhiều hơn một chút, nhưng luôn giữ mức thấp.
    clahe_weight = float(np.clip(0.16 + glare_ratio * 1.2, 0.16, 0.30))
    lightness_final = cv2.addWeighted(
        lightness_gamma, 1.0 - clahe_weight,
        lightness_clahe, clahe_weight,
        0,
    )

    corrected_lab = cv2.merge((lightness_final, channel_a, channel_b))
    processed_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    if severe_glare_ratio >= 0.10 or glare_ratio >= 0.22:
        quality = "severe_glare"
    elif severe_glare_ratio >= 0.035 or glare_ratio >= 0.08:
        quality = "high_glare"
    else:
        quality = "corrected"

    return processed_bgr, {
        "corrected": True,
        "quality": quality,
        "mean_brightness": round(mean_v, 2),
        "median_brightness": round(median_v, 2),
        "p98_brightness": round(p98_v, 2),
        "glare_ratio": round(glare_ratio, 4),
        "severe_glare_ratio": round(severe_glare_ratio, 4),
        "gamma": round(gamma, 3),
        "white_scale": round(white_scale, 3),
    }


def cnn_predict(crop_bgr):
    # Lớp bảo vệ cuối: mọi ảnh đưa vào CNN luôn là hình vuông trước khi resize.
    square_crop = center_crop_square(crop_bgr)

    # Preprocessing chỉ dùng cho CNN. Ảnh gốc vẫn được giữ cho UI và model .pt.
    processed_crop, lighting_info = preprocess_food_lighting(square_crop)

    interpolation = cv2.INTER_AREA if processed_crop.shape[0] >= IMG_HEIGHT else cv2.INTER_CUBIC
    resized = cv2.resize(processed_crop, (IMG_WIDTH, IMG_HEIGHT), interpolation=interpolation)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    global_pred = global_model.predict(arr, verbose=0)[0]
    top2_idx = np.argsort(global_pred)[::-1][:2]
    top2_group = [(GLOBAL_CLASSES[i], float(global_pred[i])) for i in top2_idx]

    final_results = []
    for group, group_prob in top2_group:
        if group not in SUB_MODELS:
            final_results.append({"class": group, "prob": float(group_prob)})
            continue

        sub_pred = loaded_sub_models[group].predict(arr, verbose=0)[0]
        for i, cls in enumerate(SUB_MODELS[group]["classes"]):
            final_results.append({"class": cls, "prob": float(group_prob * sub_pred[i])})

    final_results.sort(key=lambda x: x["prob"], reverse=True)
    best = final_results[0]
    return best["class"], best["prob"], final_results[:5], lighting_info


def display_class_name(final_class):
    return CLASS_LABEL_ALIASES.get(final_class, final_class)


def get_static_price(final_class):
    display_name = display_class_name(final_class)
    return PRICE_TABLE.get(display_name, PRICE_TABLE.get(final_class, 0))


def get_thit_kho_price(egg_count):
    if egg_count <= 0:
        return {
            "final_class": "Thịt kho",
            "price": THIT_KHO_NO_EGG_PRICE,
            "base_price": THIT_KHO_NO_EGG_PRICE,
            "egg_extra_price": 0,
            "price_note": "Không phát hiện trứng",
        }

    extra_eggs = max(0, egg_count - EGG_BASE_COUNT)
    egg_extra_price = extra_eggs * EGG_EXTRA_PRICE
    total = THIT_KHO_EGG_BASE_PRICE + egg_extra_price
    if extra_eggs > 0:
        note = f"1 trứng mặc định + {extra_eggs} trứng thêm (+{egg_extra_price:,}đ)"
    else:
        note = "1 trứng mặc định"
    return {
        "final_class": "Thịt kho trứng",
        "price": total,
        "base_price": THIT_KHO_EGG_BASE_PRICE,
        "egg_extra_price": egg_extra_price,
        "price_note": note,
    }


def format_top5(top5):
    return [
        {"class": display_class_name(r["class"]), "prob": round(r["prob"] * 100, 2)}
        for r in top5
    ]


def bgr_to_base64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def draw_text_bg(img, text, x, y, color=(86, 28, 36)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(th + 8, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 4, y + baseline + 4), (255, 248, 239), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def yolo_egg_check(crop_bgr):
    """Kiểm tra biến thể Thịt kho trong crop. Không vẽ annotation lên ảnh trả về."""
    if yolo_model is None:
        return 0, [], yolo_error or "Bộ kiểm tra biến thể chưa sẵn sàng"

    result = yolo_model(crop_bgr, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    egg_boxes = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id != EGG_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1 = max(0, min(crop_bgr.shape[1] - 1, x1))
            y1 = max(0, min(crop_bgr.shape[0] - 1, y1))
            x2 = max(0, min(crop_bgr.shape[1] - 1, x2))
            y2 = max(0, min(crop_bgr.shape[0] - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            egg_boxes.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "conf": round(conf * 100, 2),
            })

    return len(egg_boxes), egg_boxes, ""


def normalize_layout(layout):
    return layout if layout in COORDS_PRESETS else "top3"


def get_interactive_boxes(data, img_w, img_h):
    boxes = data.get("boxes") or []
    parsed = []
    for idx, box in enumerate(boxes[:8], start=1):
        name = str(box.get("name") or f"Món {idx}")
        if box.get("unit") == "normalized":
            rect = {
                "x": float(box.get("x", 0)) * img_w,
                "y": float(box.get("y", 0)) * img_h,
                "w": float(box.get("w", 0)) * img_w,
                "h": float(box.get("h", 0)) * img_h,
            }
        else:
            rect = {
                "x": float(box.get("x", 0)),
                "y": float(box.get("y", 0)),
                "w": float(box.get("w", 0)),
                "h": float(box.get("h", 0)),
            }
        parsed.append((name, rect))
    return parsed


def process_crop(name, crop):
    # Dùng cùng một crop vuông cho CNN, kiểm tra biến thể và thumbnail.
    crop = center_crop_square(crop)
    cnn_class, cnn_prob, top5, lighting_info = cnn_predict(crop)
    display_final_class = display_class_name(cnn_class)
    price = get_static_price(cnn_class)
    base_price = price
    egg_count = 0
    egg_boxes = []
    egg_extra_price = 0
    price_note = ""
    yolo_checked = False
    yolo_error_msg = ""
    display_crop = crop

    if display_final_class == "Thịt kho":
        yolo_checked = True
        egg_count, egg_boxes, yolo_error_msg = yolo_egg_check(crop)
        price_info = get_thit_kho_price(egg_count)
        display_final_class = price_info["final_class"]
        price = price_info["price"]
        base_price = price_info["base_price"]
        egg_extra_price = price_info["egg_extra_price"]
        price_note = price_info["price_note"]
        display_crop = crop

    thumb = cv2.resize(display_crop, (220, 170), interpolation=cv2.INTER_AREA)
    top5_display = format_top5(top5)
    if display_final_class == "Thịt kho trứng" and top5_display:
        # UI-facing Top-5 should stay consistent with the final dish label.
        top5_display[0]["class"] = "Thịt kho trứng"

    return {
        "name": name,
        "final_class": display_final_class,
        "raw_class": cnn_class,
        "cnn_prob": round(cnn_prob * 100, 2),
        "price": int(price),
        "top5": top5_display,
        "thumb_b64": bgr_to_base64(thumb),
        # Metadata phục vụ kiểm thử backend; UI hiện tại không cần hiển thị.
        "lighting": lighting_info,
    }

# ── FLASK APP ─────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "CNN",
    })


@app.route("/predict", methods=["POST"])
def predict():
    started_at = time.perf_counter()
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "Thiếu dữ liệu ảnh"}), 400

        orig = decode_base64_image(data["image"])
        if orig is None:
            return jsonify({"error": "Không đọc được ảnh"}), 400

        img_h, img_w = orig.shape[:2]
        interactive_boxes = get_interactive_boxes(data, img_w, img_h)

        items = []
        used_boxes = []

        if interactive_boxes:
            for name, rect in interactive_boxes:
                crop, clipped = crop_item_rect(orig, rect)
                used_boxes.append({"name": name, **clipped})
                items.append(process_crop(name, crop))
        else:
            layout_key = normalize_layout(data.get("layout", "top3"))
            standard = resize_and_crop_center(orig, STD_W, STD_H)
            for name, pts in COORDS_PRESETS[layout_key]:
                crop = crop_item_perspective(standard, pts)
                items.append(process_crop(name, crop))

        total = sum(it["price"] for it in items)
        processing_ms = round((time.perf_counter() - started_at) * 1000, 1)

        return jsonify({
            "items": items,
            "total": int(total),
            "processing_ms": processing_ms,
            "model": "CNN",
            "source_size": {"w": img_w, "h": img_h},
            "used_boxes": used_boxes,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
