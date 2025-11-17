import os
from pathlib import Path

import numpy as np
from PIL import Image
import PySimpleGUI as sg
import cv2

try:
    import fitz  # PyMuPDF for PDF disguised as JPG
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------- "Theme" / CSS-like settings ----------
APP_TITLE = "Photo Rectangle Auto-Cropper"

# Colours (change these to whatever you like)
COLOR_BG          = "#7ea172"  # window background (Beauchamps Green) #7ea172 
COLOR_CARD        = "#4e6766"  # panel background #4e6766 
COLOR_TEXT        = "#e5e7eb"  # normal text
COLOR_ACCENT      = "#837c73"  # buttons, highlights (Oil brown) #837c73 
COLOR_ACCENT_TEXT = "#FFFFFF"  # text on accent buttons
COLOR_INPUT_BG    = "#020617"  # input fields
COLOR_INPUT_TEXT  = "#e5e7eb"
COLOR_LOG_BG      = "#020617"
COLOR_LOG_TEXT    = "#e5e7eb"

FONT_HEADER = ("Segoe UI", 14, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas", 9)

# Apply a neutral base theme; we override colours manually
sg.theme("DarkGrey5")


# -----------------------------
# LOAD IMAGE (JPEG / PNG / PDF-MASQUERADING-AS-JPEG)
# -----------------------------
def load_image_any(input_path: str) -> Image.Image:
    """
    Load an image file and return a PIL Image in RGB mode.
    Handles:
      - normal JPG/PNG
      - 'PDF' files that have been given a .jpg extension (if PyMuPDF is available)
      - OpenCV fallback for odd encodings
    """
    p = Path(input_path)
    if not p.exists():
        raise RuntimeError("File does not exist")

    if p.stat().st_size == 0:
        raise RuntimeError("File is empty (0 bytes)")

    # Peek at header to see if it's actually a PDF
    with open(input_path, "rb") as f:
        header = f.read(4)

    if header.startswith(b"%PDF"):
        if not HAS_FITZ:
            raise RuntimeError(
                "File appears to be a PDF but PyMuPDF is not installed. "
                "Install with: pip install pymupdf"
            )
        doc = fitz.open(input_path)
        if doc.page_count == 0:
            raise RuntimeError("PDF has no pages")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    # Try Pillow first
    try:
        img = Image.open(input_path)
        img.load()
        return img.convert("RGB")
    except Exception:
        pass

    # Fallback: OpenCV
    try:
        data = np.fromfile(input_path, dtype=np.uint8)
        cv_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise RuntimeError("cv2.imdecode returned None")
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img)
    except Exception:
        pass

    raise RuntimeError("Unable to decode image with Pillow or OpenCV")


# -----------------------------
# MAIN RECTANGLE DETECTION
# -----------------------------
def detect_main_rectangle_crop(pil_img: Image.Image) -> Image.Image:
    """
    Detect the main rectangular object (photo+frame) and crop to it.

    Strategy:
      - convert to grayscale
      - blur to reduce noise
      - Canny edges + dilation
      - find external contours
      - take the largest contour by area
      - crop to its bounding rectangle, with a small margin
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Smooth noise a bit
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Dilate edges so contours are more solid
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No contours found – just return original
        return pil_img

    # Choose largest contour by area
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add a small margin so we don't clip the frame
    margin = 10
    img_h, img_w = gray.shape

    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_w - x, w + margin * 2)
    h = min(img_h - y, h + margin * 2)

    crop_box = (x, y, x + w, y + h)

    # If the crop box is basically the whole image, don't bother
    full_box = (0, 0, img_w, img_h)
    if crop_box == full_box:
        return pil_img

    return pil_img.crop(crop_box)


# -----------------------------
# SAVE WITH CORRECT MODE
# -----------------------------
def save_image(pil_img: Image.Image, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        pil_img.convert("RGB").save(path, quality=95)
    else:
        pil_img.save(path)


# -----------------------------
# PIPELINE FOR ONE FILE
# -----------------------------
def process_image(input_path: str, output_path: str) -> bool:
    try:
        src_img = load_image_any(input_path)
        cropped = detect_main_rectangle_crop(src_img)
        save_image(cropped, output_path)
        print(f"[CROPPED] {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] {input_path}: {e}")
        return False


# -----------------------------
# GUI
# -----------------------------
def build_gui():
    # Main layout “card”
    file_section = [
        [sg.Text("Select image files", font=FONT_HEADER, text_color=COLOR_TEXT, background_color=COLOR_CARD)],
        [
            sg.Input(
                key="-FILES-",
                font=FONT_BODY,
                background_color=COLOR_INPUT_BG,
                text_color=COLOR_INPUT_TEXT,
                border_width=0,
                size=(60, 1)
            ),
            sg.FilesBrowse(
                "Browse…",
                file_types=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")),
                button_color=(COLOR_ACCENT_TEXT, COLOR_ACCENT),
                font=FONT_BODY
            ),
        ],
        [sg.Text("Output folder (if not overwriting):", font=FONT_BODY, text_color=COLOR_TEXT, background_color=COLOR_CARD)],
        [
            sg.Input(
                key="-OUT-",
                font=FONT_BODY,
                background_color=COLOR_INPUT_BG,
                text_color=COLOR_INPUT_TEXT,
                border_width=0,
                size=(60, 1)
            ),
            sg.FolderBrowse(
                "Browse…",
                button_color=(COLOR_ACCENT_TEXT, COLOR_ACCENT),
                font=FONT_BODY
            ),
        ],
        [
            sg.Checkbox(
                "Overwrite original files (otherwise saves as *_cropped.ext)",
                key="-OVERWRITE-",
                default=False,
                font=FONT_BODY,
                text_color=COLOR_TEXT,
                background_color=COLOR_CARD,
            )
        ],
    ]

    controls_section = [
        [
            sg.Button(
                "Process",
                key="-PROCESS-",
                size=(12, 1),
                font=FONT_BODY,
                button_color=(COLOR_ACCENT_TEXT, COLOR_ACCENT),
                border_width=0,
            ),
            sg.Button(
                "Exit",
                size=(8, 1),
                font=FONT_BODY,
                button_color=(COLOR_TEXT, "#4b5563"),
                border_width=0,
            ),
        ]
    ]

    log_section = [
        [sg.Text("Log", font=FONT_HEADER, text_color=COLOR_TEXT, background_color=COLOR_CARD)],
        [
            sg.Multiline(
                "",
                size=(90, 20),
                key="-LOG-",
                disabled=True,
                autoscroll=True,
                font=FONT_MONO,
                background_color=COLOR_LOG_BG,
                text_color=COLOR_LOG_TEXT,
                border_width=0,
            )
        ],
    ]

    layout = [
        [
            sg.Column(
                [
                    [sg.Text(APP_TITLE, font=("Segoe UI Semibold", 16), text_color=COLOR_TEXT, background_color=COLOR_BG)],
                    [sg.Text("Automatically crop scanned photos from the flatbed area.", font=FONT_BODY, text_color="#ffffff", background_color=COLOR_BG)],
                    [sg.VPush()],
                    [
                        sg.Frame(
                            "",
                            file_section + controls_section,
                            background_color=COLOR_CARD,
                            pad=(0, 10),
                            border_width=0,
                        )
                    ],
                    [
                        sg.Frame(
                            "",
                            log_section,
                            background_color=COLOR_CARD,
                            pad=(0, 10),
                            border_width=0,
                        )
                    ],
                    [sg.VPush()],
                ],
                background_color=COLOR_BG,
                pad=(15, 15),
            )
        ]
    ]

    return sg.Window(
        APP_TITLE,
        layout,
        finalize=True,
        background_color=COLOR_BG,
        element_justification="left",
    )


def main():
    window = build_gui()

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Process":
            files_str = values["-FILES-"].strip()
            out_dir = values["-OUT-"].strip()
            overwrite = values["-OVERWRITE-"]

            log = window["-LOG-"]
            log.update("")

            if not files_str:
                log.update("Please select at least one image file.\n")
                continue

            file_paths = [f for f in files_str.split(";") if f.strip()]
            file_paths = [
                f
                for f in file_paths
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            ]

            if not file_paths:
                log.update("No supported image files selected.\n")
                continue

            if not overwrite and not out_dir:
                log.update(
                    "Please select an output folder or tick 'Overwrite original files'.\n"
                )
                continue

            if overwrite:
                log.update("WARNING: Overwriting original files.\n\n")
            else:
                log.update(f"Output folder: {out_dir}\n\n")

            processed = 0
            for src in file_paths:
                src = src.strip()
                base = os.path.basename(src)
                name, ext = os.path.splitext(base)

                if overwrite:
                    dest = src
                else:
                    dest = os.path.join(out_dir, f"{name}_cropped{ext}")

                ok = process_image(src, dest)
                if ok:
                    processed += 1
                    log.update(log.get() + f"✓ {base} -> {os.path.basename(dest)}\n")
                else:
                    log.update(log.get() + f"✗ Failed on {base}\n")

            log.update(
                log.get()
                + f"\nDone. Successfully processed {processed} of {len(file_paths)} file(s).\n"
            )

    window.close()


if __name__ == "__main__":
    main()
