"""
Interactive webcam data collector for sentence-level sign-language clips.

Camera stays open. You press a key (1/2/3 or A/B/C) to choose which sentence
you are about to sign, then press 'r' to record. The clip is saved to that
sentence's folder. After each clip you choose a (possibly new) sentence again.

    dataset/<sentence_name>/, e.g. dataset/Khushi_zindagi_mein_ahem_hai/
    dataset/<sentence_name>/<sentence_name>_<clip_idx>.mp4

Usage:
    python collect_sign_dataset.py

Dependencies:
    pip install opencv-python
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

# Sentences to capture (Urdu); extend this list to add more (keep SENTENCE_DISPLAY in sync).
# Keys A–T map to the folders currently under dataset/ (original order, gaps removed).
SENTENCES: List[str] = [
    "کھڑکی بند کرو",                        # A
    "آپ کیسے ہیں",                          # B
    "کتنی دیر",                             # C
    "مجھے سمجھ نہیں آتی",                   # D
    "چلے جاؤ",                              # E
    "ڈرو مت",                               # F
    "شکریہ",                                # G
    "آپ کا نام کیا ہے",                     # H
    "کیا آپ تیار ہیں",                      # I
    "میں بیمار ہوں",                        # J
    "کیا آپ نے کھانا کھایا",               # K
    "عید مبارک",                            # L
    "مجھے اپنی قمیص دھونی ہے",              # M
    "میں اشاروں کی زبان سیکھ رہا ہوں",      # N
    "ہاتھ مت لگاؤ",                         # O
    "ادھر آو!",                             # P
    "میں جلد ہی واپس آؤں گا.",               # Q
    "معاف کیجئے گا.",                       # R
    "مجھے خریداری کرنے جانا ہے",            # S
    "کیا میں آپ کی مدد کر سکتا ہوں",         # T
]
# English labels for overlays / clip filenames (same order as SENTENCES).
SENTENCE_DISPLAY: List[str] = [
    "close the window",              # A
    "how are you",                   # B
    "how long",                      # C
    "i do not understand",           # D
    "go away",                       # E
    "dont be afraid",                # F
    "thank you",                     # G
    "what is your name",             # H
    "are you ready",                 # I
    "i am sick",                     # J
    "did you eat",                   # K
    "eid mubarak",                   # L
    "i need to wash my shirt",       # M
    "i am learning sign language",   # N
    "dont touch",                    # O
    "come here",                     # P
    "i will be back soon",           # Q
    "excuse me",                     # R
    "i have to go shopping",         # S
    "can i help you",                # T
]


def _build_key_to_sentence(num_sentences: int) -> Dict[int, int]:
    """Map keyboard codes to sentence index.
    - Uppercase A–Z(i) and lowercase a–z(i) for each sentence, **except** lowercase
      ``q`` and ``r`` — those are reserved for quit / start-recording.
    """
    m: Dict[int, int] = {}
    for i in range(min(num_sentences, 26)):
        upper = chr(ord("A") + i)
        lower = chr(ord("a") + i)
        m[ord(upper)] = i
        if lower not in ("q", "r"):
            m[ord(lower)] = i
    return m


KEY_TO_SENTENCE = _build_key_to_sentence(len(SENTENCES))
# Seconds captured per clip (longer for full sentences).
CLIP_DURATION_SECONDS = 45
# FPS target for both preview and saved clips.
TARGET_FPS = 20.0
# Lazily detected from the first frame so we match webcam resolution.
FRAME_SIZE: Tuple[int, int] | None = None
# Cached ROI output size (width, height) used for saving the cropped video.
ROI_FRAME_SIZE: Tuple[int, int] | None = None
# ROI bounds expressed as fractions of width/height: (x1, y1, x2, y2).
ROI_RELATIVE = (0.25, 0.15, 0.75, 0.85)
ROI_COLOR = (0, 0, 255)  # BGR red box to draw attention.
# Root dataset directory; each sentence gets a subfolder named after the Urdu sentence.
DATASET_ROOT = Path("dataset")


def _safe_folder_name(text: str) -> str:
    """Make a filesystem-safe folder name from sentence text (spaces -> underscores, remove bad chars)."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)
    return safe.strip("_").replace(" ", "_") or "sentence"


def sentence_id(sentence_index: int) -> str:
    """Return folder name for the sentence at the given index (same as Urdu sentence text, sanitized)."""
    sentence_text = SENTENCES[sentence_index]
    return _safe_folder_name(sentence_text)


def sentence_display(sentence_index: int) -> str:
    """Return display text for overlay; falls back to 'Sentence N' if not in SENTENCE_DISPLAY."""
    if sentence_index < len(SENTENCE_DISPLAY):
        return SENTENCE_DISPLAY[sentence_index]
    return f"Sentence {sentence_index}"


def ensure_dirs() -> None:
    """Create dataset directories for every sentence and save sentence text."""
    for i in range(len(SENTENCES)):
        folder = DATASET_ROOT / sentence_id(i)
        folder.mkdir(parents=True, exist_ok=True)
        # Save the sentence text for reference (e.g. for training)
        (folder / "sentence.txt").write_text(SENTENCES[i], encoding="utf-8")


def get_next_clip_index(sentence_index: int) -> int:
    """Return the next clip number (1-based) by finding max existing number in our filename pattern."""
    stem = _safe_folder_name(sentence_display(sentence_index)) or "clip"
    sentence_dir = DATASET_ROOT / sentence_id(sentence_index)
    if not sentence_dir.exists():
        return 1
    pattern = re.compile(re.escape(stem) + r"_(\d+)\.(?:avi|mp4)$", re.IGNORECASE)
    max_num = 0
    for f in sentence_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def run_preview(
    cap: cv2.VideoCapture,
    selected_sentence: int | None,
) -> Tuple[bool, int | None]:
    """Show continuous webcam preview. User selects sentence (A–…; lowercase q/r reserved), then 'r' to record or 'q' to quit.
    Returns (True, sentence_index) to record for that sentence, or (False, None) to quit.
    """
    global FRAME_SIZE

    ret, frame = cap.read()
    if not ret:
        return False, None

    height, width = frame.shape[:2]
    if FRAME_SIZE is None:
        FRAME_SIZE = (width, height)

    x1, y1, x2, y2 = _absolute_roi()

    n = len(SENTENCES)
    last_letter = chr(ord("A") + n - 1) if n <= 26 else "Z"
    print(
        f"[INFO] Press A-{last_letter} to select (lowercase ok; for Q/R use Shift+Q / Shift+R), "
        "then 'r' to record. 'q' to quit."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), ROI_COLOR, 2)

        if selected_sentence is not None:
            eng = sentence_display(selected_sentence)
            eng_label = (eng[:40] + "..") if len(eng) > 40 else eng
            cv2.putText(
                overlay,
                f"[{chr(ord('A') + selected_sentence)}] {eng_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                f"No sentence - press A-{last_letter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            f"select: A-{last_letter} (q/r=quit/rec) | r=rec | q=quit",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Position yourself in the red box, select sentence, then press r",
            (10, FRAME_SIZE[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sign Dataset Capture - Preview", overlay)
        key = cv2.waitKey(1) & 0xFF

        # Quit / record must win over letter Q/R sentence selection
        if key == ord("q"):
            cv2.destroyWindow("Sign Dataset Capture - Preview")
            return False, None
        if key == ord("r"):
            if selected_sentence is not None:
                cv2.destroyWindow("Sign Dataset Capture - Preview")
                return True, selected_sentence
            print(f"[WARN] Select a sentence first (A-{last_letter}; Q/R = Shift+Q/R).")
        elif key in KEY_TO_SENTENCE:
            idx = KEY_TO_SENTENCE[key]
            if idx < len(SENTENCES):
                selected_sentence = idx
                print(f"[SELECT] {chr(ord('A') + idx)}: {SENTENCES[idx]}  [{sentence_display(idx)}]")

        if cv2.getWindowProperty("Sign Dataset Capture - Preview", cv2.WND_PROP_VISIBLE) < 1:
            try:
                user_input = input(
                    f"Preview closed. Letter A-{last_letter} (Q/R = capital), or number 1-{len(SENTENCES)}, or 'q' quit: "
                ).strip()
                if user_input.lower() == "q":
                    return False, None
                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(SENTENCES):
                        selected_sentence = idx
                        print(f"[SELECT] {chr(ord('A') + idx)}: {SENTENCES[idx]}  [{sentence_display(idx)}]")
                elif len(user_input) == 1:
                    ch = user_input
                    idx = -1
                    if "A" <= ch <= last_letter:
                        idx = ord(ch) - ord("A")
                    elif "a" <= ch <= last_letter.lower():
                        if ch in ("q", "r"):
                            print("[WARN] For Q/R use Shift (capital Q/R) or type 17 / 18.")
                        else:
                            idx = ord(ch) - ord("a")
                    if 0 <= idx < len(SENTENCES):
                        selected_sentence = idx
                        print(f"[SELECT] {chr(ord('A') + idx)}: {SENTENCES[idx]}  [{sentence_display(idx)}]")
            except (EOFError, KeyboardInterrupt):
                return False, None


def capture_clip(cap: cv2.VideoCapture, sentence_index: int, clip_index: int) -> None:
    """Capture a short video clip for the specified sentence."""
    global FRAME_SIZE, ROI_FRAME_SIZE

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Unable to read from webcam. Is it already in use?")

    height, width = frame.shape[:2]
    if FRAME_SIZE is None:
        FRAME_SIZE = (width, height)
    x1, y1, x2, y2 = _absolute_roi()
    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI configuration yields non-positive dimensions.")
    if ROI_FRAME_SIZE is None:
        ROI_FRAME_SIZE = (roi_width, roi_height)

    sid = sentence_id(sentence_index)
    sentence_dir = DATASET_ROOT / sid
    sentence_dir.mkdir(parents=True, exist_ok=True)

    # Use ASCII-safe filename (keep Urdu folder name only).
    stem = _safe_folder_name(sentence_display(sentence_index)) or "clip"

    # Try a few common container/codec combos; MJPG/AVI is usually safest on macOS.
    attempts = [
        (".avi", "MJPG"),
        (".avi", "XVID"),
        (".mp4", "mp4v"),
    ]
    writer = None
    filepath = None
    tried = []
    for ext, codec_str in attempts:
        candidate = sentence_dir / f"{stem}_{clip_index}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        test_writer = cv2.VideoWriter(str(candidate), fourcc, TARGET_FPS, FRAME_SIZE)
        tried.append(f"{codec_str} -> {candidate.name}")
        if test_writer.isOpened():
            writer = test_writer
            filepath = candidate
            break
        test_writer.release()

    if writer is None or filepath is None:
        raise RuntimeError(
            "Failed to open video writer for any format. Tried: "
            + ", ".join(tried)
        )

    start_time = time.time()
    frames_needed = int(CLIP_DURATION_SECONDS * TARGET_FPS)

    frames_written = 0
    while frames_written < frames_needed:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed; stopping clip early.")
            break

        overlay = frame.copy()
        # Draw quick visual cues onto the preview window.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), ROI_COLOR, 2)
        eng = sentence_display(sentence_index)
        eng_label = (eng[:40] + "..") if len(eng) > 40 else eng
        cv2.putText(
            overlay,
            f"[{chr(ord('A') + sentence_index)}] {eng_label} (clip {clip_index})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        remaining = max(0, CLIP_DURATION_SECONDS - (time.time() - start_time))
        cv2.putText(
            overlay,
            f"Recording... {remaining:0.1f}s left",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Keep hands inside the red box",
            (10, FRAME_SIZE[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sign Dataset Capture", overlay)
        writer.write(frame)  # Persist full frame.
        frames_written += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Recording aborted early via 'q'.")
            break

    writer.release()
    print(f"[OK] Saved {filepath} ({frames_written} frames).")


def main() -> None:
    ensure_dirs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam 0.")

    n = len(SENTENCES)
    last_letter = chr(ord("A") + n - 1) if n <= 26 else "Z"
    print("Sign-language dataset capture (sentence-level)")
    print(
        f"Keys: A-{last_letter} → select (note: Q/R = Shift+Q / Shift+R; plain q,r = quit/record); "
        f"r = record; clip → Urdu folder."
    )
    print("After each clip, pick the sentence again. Press q to quit.")
    print("Sentences:")
    for i, s in enumerate(SENTENCES):
        letter = chr(ord("A") + i) if i < 26 else "?"
        print(f"  {letter} ({i + 1}): {s}  [{sentence_display(i)}]")

    selected_sentence: int | None = None

    try:
        while True:
            wants_record, sentence_index = run_preview(cap, selected_sentence)
            if not wants_record or sentence_index is None:
                print("[EXIT] Quit.")
                break

            clip_index = get_next_clip_index(sentence_index)
            print(f"[INFO] Recording [{chr(ord('A') + sentence_index)}] → clip {clip_index}...")
            capture_clip(cap, sentence_index, clip_index)

            # Next recording: user must select a sentence again
            selected_sentence = None

    finally:
        cap.release()
        cv2.destroyAllWindows()


def _absolute_roi() -> Tuple[int, int, int, int]:
    """Convert relative ROI config to absolute pixel coordinates."""
    if FRAME_SIZE is None:
        raise RuntimeError("FRAME_SIZE must be initialized before computing ROI.")
    width, height = FRAME_SIZE
    rel_x1, rel_y1, rel_x2, rel_y2 = ROI_RELATIVE
    x1 = int(width * rel_x1)
    y1 = int(height * rel_y1)
    x2 = int(width * rel_x2)
    y2 = int(height * rel_y2)
    return x1, y1, x2, y2


if __name__ == "__main__":
    main()

