"""
services/media_analyzer.py
============================
Image and video analysis for DataSense AI, following the same pattern as
services/data_quality.py and services/dashboard_builder.py: given a file,
return the numbers the UI needs. Here that's a (kpis, charts) pair using
the exact widget schema dashboard.html already renders (bar, line, donut,
gauge, table, stacked_bar, card - see the comment block in that template's
<script> tag for the full list).

analyze_image(filepath) -> (kpis, charts)
analyze_video(filepath) -> (kpis, charts)
"""

import os
from collections import Counter

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------
def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(max(0, min(255, c))) for c in rgb)


def _format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------
# IMAGE ANALYSIS
# ---------------------------------------------------------------------
def _dominant_colors(img_rgb, k=5):
    """K-means color quantization to find the k most dominant colors."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    if pixels.shape[0] > 20000:
        idx = np.random.choice(pixels.shape[0], 20000, replace=False)
        pixels = pixels[idx]

    k = max(1, min(k, len(np.unique(pixels.astype(int), axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    counts = Counter(labels.flatten().tolist())
    total = sum(counts.values())
    results = [
        {"hex": _rgb_to_hex(centers[idx]), "pct": round(count / total * 100, 1)}
        for idx, count in counts.most_common()
    ]
    results.sort(key=lambda c: c["pct"], reverse=True)
    return results


def _blur_score(gray):
    """Variance of the Laplacian - higher = sharper, lower = blurrier."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return round((np.count_nonzero(edges) / edges.size) * 100, 2)


def analyze_image(image_path):
    """Analyze a single image file. Returns (kpis, charts)."""
    file_size_kb = round(os.path.getsize(image_path) / 1024, 1)

    pil_img = Image.open(image_path)
    fmt = pil_img.format or "Unknown"
    width, height = pil_img.size
    aspect_ratio = round(width / height, 2) if height else 0

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    brightness = round(float(np.mean(gray)) / 255 * 100, 1)
    contrast = round(float(np.std(gray)), 1)
    saturation = round(float(np.mean(hsv[:, :, 1])) / 255 * 100, 1)
    sharpness = round(float(_blur_score(gray)), 1)
    edges_pct = _edge_density(gray)

    dominant = _dominant_colors(img_rgb, k=5)

    bins = 16
    hist_labels = [str(i * (256 // bins)) for i in range(bins)]
    r_hist, _ = np.histogram(img_rgb[:, :, 0], bins=bins, range=(0, 256))
    g_hist, _ = np.histogram(img_rgb[:, :, 1], bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(img_rgb[:, :, 2], bins=bins, range=(0, 256))

    kpis = [
        {"icon": "📐", "value": f"{width}×{height}", "label": "Resolution"},
        {"icon": "💾", "value": f"{file_size_kb} KB", "label": "File Size"},
        {"icon": "☀️", "value": f"{brightness}%", "label": "Brightness"},
        {"icon": "🎯", "value": f"{sharpness:.0f}", "label": "Sharpness Score"},
        {"icon": "🎨", "value": dominant[0]["hex"].upper(), "label": "Dominant Color"},
        {"icon": "🖼️", "value": fmt, "label": "Format"},
    ]

    charts = [
        {
            "type": "stacked_bar",
            "title": "RGB Channel Distribution",
            "labels": hist_labels,
            "series": [
                {"name": "Red", "values": r_hist.tolist()},
                {"name": "Green", "values": g_hist.tolist()},
                {"name": "Blue", "values": b_hist.tolist()},
            ],
        },
        {
            "type": "donut",
            "title": "Dominant Color Palette",
            "labels": [c["hex"] for c in dominant],
            "values": [c["pct"] for c in dominant],
        },
        {"type": "gauge", "title": "Brightness", "value": brightness, "max": 100, "suffix": "%"},
        {"type": "gauge", "title": "Saturation", "value": saturation, "max": 100, "suffix": "%"},
        {
            "type": "card",
            "title": "Edge Density",
            "value": f"{edges_pct}%",
            "label": "Of pixels are edges (detail level)",
        },
        {
            "type": "table",
            "title": "Image Properties",
            "columns": ["Property", "Value"],
            "rows": [
                ["Format", fmt],
                ["Dimensions", f"{width} × {height} px"],
                ["Aspect Ratio", f"{aspect_ratio}:1"],
                ["File Size", f"{file_size_kb} KB"],
                ["Color Mode", pil_img.mode],
                ["Contrast (std dev)", contrast],
            ],
        },
    ]

    return kpis, charts


# ---------------------------------------------------------------------
# VIDEO THUMBNAIL (for <video poster> - lets the browser show a preview
# frame instantly, without downloading any actual video data)
# ---------------------------------------------------------------------
def extract_video_thumbnail(video_path, output_path, at_fraction=0.1):
    """Grabs a representative frame from a video and saves it as a JPEG.
    Picks a frame ~10% into the clip by default rather than frame 0, since
    the very first frame is often a black/fade-in frame. Returns True on
    success, False if no frame could be read at all."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(total_frames * at_fraction) if total_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()

    if not ret:
        # fall back to the very first frame if seeking ahead failed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()

    if not ret:
        return False

    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


# ---------------------------------------------------------------------
# VIDEO ANALYSIS
# ---------------------------------------------------------------------
def analyze_video(video_path, sample_fps=2, scene_change_threshold=35.0):
    """Analyze a video file by sampling frames. Returns (kpis, charts)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0
    file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 1)

    frame_interval = max(int(fps / sample_fps), 1) if fps else 1

    timestamps, brightness_series, motion_series = [], [], []
    scene_changes = []
    color_votes = Counter()

    prev_gray = None
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(frame, (80, 45))
            timestamp = frame_idx / fps if fps else 0

            brightness = float(np.mean(gray)) / 255 * 100
            timestamps.append(_format_timestamp(timestamp))
            brightness_series.append(round(brightness, 1))

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(np.mean(diff))
                motion_series.append(round(motion, 1))
                if motion > scene_change_threshold:
                    scene_changes.append(_format_timestamp(timestamp))
            else:
                motion_series.append(0.0)

            avg_color = small.reshape(-1, 3).mean(axis=0)[::-1]  # BGR -> RGB
            bucketed = tuple(int(c // 32 * 32) for c in avg_color)
            color_votes[bucketed] += 1

            prev_gray = gray
            sampled += 1

        frame_idx += 1

    cap.release()

    avg_brightness = round(float(np.mean(brightness_series)), 1) if brightness_series else 0
    avg_motion = round(float(np.mean(motion_series)), 1) if motion_series else 0

    top_colors = color_votes.most_common(5)
    total_votes = sum(count for _, count in top_colors) or 1
    palette = [
        {"hex": _rgb_to_hex(color), "pct": round(count / total_votes * 100, 1)}
        for color, count in top_colors
    ]

    kpis = [
        {"icon": "⏱️", "value": _format_timestamp(duration), "label": "Duration"},
        {"icon": "🎞️", "value": f"{fps:.0f} fps", "label": "Frame Rate"},
        {"icon": "📐", "value": f"{width}×{height}", "label": "Resolution"},
        {"icon": "💾", "value": f"{file_size_mb} MB", "label": "File Size"},
        {"icon": "🎬", "value": len(scene_changes), "label": "Scene Cuts Detected"},
        {"icon": "🏃", "value": f"{avg_motion}", "label": "Avg. Motion Level"},
    ]

    charts = [
        {"type": "line", "title": "Brightness Over Time", "labels": timestamps, "values": brightness_series},
        {"type": "area", "title": "Motion / Activity Over Time", "labels": timestamps, "values": motion_series},
        {
            "type": "donut",
            "title": "Dominant Colors Across Video",
            "labels": [c["hex"] for c in palette],
            "values": [c["pct"] for c in palette],
        },
        {"type": "gauge", "title": "Average Brightness", "value": avg_brightness, "max": 100, "suffix": "%"},
        {
            "type": "table",
            "title": "Detected Scene Changes",
            "columns": ["#", "Timestamp"],
            "rows": [[i + 1, t] for i, t in enumerate(scene_changes)] or [["-", "No hard cuts detected"]],
        },
        {
            "type": "table",
            "title": "Video Properties",
            "columns": ["Property", "Value"],
            "rows": [
                ["Duration", _format_timestamp(duration)],
                ["Frame Rate", f"{fps:.1f} fps"],
                ["Total Frames", total_frames],
                ["Resolution", f"{width} × {height} px"],
                ["File Size", f"{file_size_mb} MB"],
                ["Frames Sampled", sampled],
            ],
        },
    ]

    return kpis, charts