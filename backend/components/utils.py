import numpy as np
import cv2, tempfile, os

def draw_box_np(img_rgb, box, color=(255, 0, 0), thickness=3):
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(img_rgb, (x0, y0), (x1, y1), color, thickness)

def draw_box_with_label_np(img_rgb, box, text, color=(255, 0, 0), thickness=3):
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(img_rgb, (x0, y0), (x1, y1), color, thickness)
    t = text or ""
    if not t:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    th = 2
    (tw, th_text), _ = cv2.getTextSize(t, font, scale, th)
    pad = 5
    bx0, by0 = x0, max(0, y0 - (th_text + 2 * pad) - 2)
    bx1, by1 = x0 + tw + 2 * pad, y0
    cv2.rectangle(img_rgb, (bx0, by0), (bx1, by1), color, -1)
    cv2.putText(img_rgb, t, (x0 + pad, y0 - pad), font, scale, (255, 255, 255), th, cv2.LINE_AA)

def draw_red_spot_np(img_rgb, center, radius=14):
    if center is None:
        return
    x, y = map(int, center)
    cv2.circle(img_rgb, (x, y), max(2, radius // 2), (0, 0, 255), -1)
    cv2.circle(img_rgb, (x, y), radius, (0, 0, 255), 2)

def overlay_heatmap_in_box(img_rgb, box, heatmap_crop, alpha=0.5):
    """heatmap_crop: np.float32 (Hc,Wc) in [0,1] — sẽ resize vào bbox và apply COLORMAP_JET"""
    x0, y0, x1, y1 = map(int, box)
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(img_rgb.shape[1], x1); y1 = min(img_rgb.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    H, W = y1 - y0, x1 - x0
    hm = (np.clip(heatmap_crop, 0, 1) * 255.0).astype(np.uint8)
    hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]  # -> RGB
    roi = img_rgb[y0:y1, x0:x1, :]
    blended = cv2.addWeighted(roi, 1.0, hm_color, alpha, 0)
    img_rgb[y0:y1, x0:x1, :] = blended

def draw_saliency_dots_in_box(img_rgb, box, heatmap_crop, density=0.02):
    """
    Rải chấm đỏ theo saliency: lấy top 'density' pixels (mặc định 2%) trong bbox.
    """
    x0, y0, x1, y1 = map(int, box)
    H, W = max(1, y1 - y0), max(1, x1 - x0)
    hm = cv2.resize(np.clip(heatmap_crop, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)
    thr = np.quantile(hm, 1.0 - max(0.001, min(0.2, density)))
    ys, xs = np.where(hm >= thr)
    # Lấy bớt điểm cho gọn (grid step)
    if xs.size > 0:
        step = max(1, int(0.02 * max(H, W)))
        keep = (xs % step == 0) & (ys % step == 0)
        xs, ys = xs[keep], ys[keep]
        for x, y in zip(xs + x0, ys + y0):
            cv2.circle(img_rgb, (int(x), int(y)), 3, (0, 0, 255), -1)

def _render_fake_real_bar(p_fake, p_real):
    pf = max(0, min(100, int(round(p_fake * 100))))
    pr = max(0, min(100, int(round(p_real * 100))))
    w = 300
    w_f = int(w * pf / 100.0); w_r = w - w_f
    html = f"""
    <div style="display:flex;gap:8px;align-items:center">
      <div style="width:{w}px;height:16px;border-radius:8px;overflow:hidden;border:1px solid #999;display:flex">
        <div style="width:{w_f}px;background:#d33"></div>
        <div style="width:{w_r}px;background:#3b7"></div>
      </div>
      <div><b>Fake</b> {pf}% &nbsp;|&nbsp; <b>Real</b> {pr}%</div>
    </div>
    """
    return html

def _make_method_table(pm, method_names):
    pm = np.asarray(pm)
    order = np.argsort(-pm)
    return [[method_names[i], round(float(pm[i] * 100.0), 1)] for i in order]

def parse_thr_map(s: str, method_names):
    out = {}
    if not s:
        return out
    for it in s.split(","):
        it = it.strip()
        if not it or "=" not in it:
            continue
        k, v = it.split("=", 1)
        k = k.strip(); v = v.strip()
        try:
            vf = float(v)
        except:
            continue
        for name in method_names:
            if name.lower() == k.lower():
                out[name] = vf
                break
    return out

def get_thr_for_method(global_thr: float, method_name: str, thr_map: dict, per_method_on: bool, gate_ok: bool):
    if per_method_on and gate_ok and method_name in thr_map:
        return float(thr_map[method_name])
    return float(global_thr)
def save_bytes_to_temp(data: bytes, suffix=".mp4") -> str:
    """
    Lưu bytes upload tạm thời vào ổ đĩa (trong thư mục hệ thống).
    Trả về đường dẫn file.
    """
    tmpdir = tempfile.mkdtemp(prefix="df_upload_")
    fname = os.path.join(tmpdir, "upload" + (suffix if suffix else ".bin"))
    with open(fname, "wb") as f:
        f.write(data)
    return fname

