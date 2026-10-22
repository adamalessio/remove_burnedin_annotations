"""
===============================================================================
DICOM Remove Image Annotations & Processing Pipeline
===============================================================================

Author:
    Adam Alessio, aalessio@msu.edu, 10/22/2026

    Coding support from Gemini, accessed 10/22/2026
    Coding support from GitHub Copilot, accessed 10/22/2026


Notice / Disclaimer:
    FOR RESEARCH PURPOSES ONLY.
    
    This software is developed strictly for non-clinical research and evaluation
    purposes. It is not intended, designed, or cleared for clinical diagnosis,
    patient treatment, or deployment in operational healthcare environments.
    Use of this code in clinical workflows is strictly prohibited.
===============================================================================
"""

# The following libraries are required for text detection and image processing. Ensure they are installed in your Python environment:
# pip install pydicom numpy opencv-python easyocr
# pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import ExplicitVRLittleEndian

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    print("[WARNING] EasyOCR not found. Falling back to OpenCV contour-based text detection. Install EasyOCR for better accuracy: pip install easyocr")
    exit(1)  # Exit if EasyOCR is not installed, as it's optimal for text detection; Can comment this out if you want to use OpenCV fallback instead.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DICOMAnonymizer")
logger.setLevel(logging.DEBUG)  # Overrides root level for this logger


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser to output usage/help message when required arguments are missing."""
    
    def error(self, message: str):
        sys.stderr.write(f"\n[ERROR] {message}\n\n")
        self.print_help(sys.stderr)
        sys.exit(2)


class TextAnnotationDetector:
    """Detects text annotations within image arrays using EasyOCR or OpenCV contour analysis."""

    def __init__(self, use_gpu: bool = False, languages: List[str] = None):
        self.use_gpu = use_gpu
        self.languages = languages or ['en']
        self.ocr_reader = None

        if HAS_EASYOCR:
            logger.info("Initializing EasyOCR text detection engine...")
            self.ocr_reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        else:
            logger.warning("EasyOCR not found. Falling back to OpenCV contour-based text heuristic.")

    def _normalize_to_uint8(self, img_array: np.ndarray) -> np.ndarray:
        """Normalizes any pixel array bit-depth (e.g., 12-bit/16-bit DICOM) to 8-bit unsigned uint8."""
        if img_array.dtype == np.uint8:
            return img_array

        img_min, img_max = img_array.min(), img_array.max()
        if img_max == img_min:
            return np.zeros(img_array.shape, dtype=np.uint8)

        normalized = ((img_array - img_min) / (img_max - img_min) * 255.0)
        return normalized.astype(np.uint8)

    def detect_text_boxes(self, img_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects text region bounding boxes in a 2D image array.
        Returns:
            List of tuples: [(ymin, xmin, ymax, xmax), ...]
        """
        norm_img = self._normalize_to_uint8(img_array)
        
        # Convert single channel grayscale to 3-channel for OCR engine if necessary
        if norm_img.ndim == 2:
            ocr_input = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
        else:
            ocr_input = norm_img

        bboxes = []

        if self.ocr_reader:
            results = self.ocr_reader.readtext(ocr_input)
            for (bbox_pts, text, prob) in results:
                pts = np.array(bbox_pts, dtype=np.int32)
                xmin = int(np.min(pts[:, 0]))
                xmax = int(np.max(pts[:, 0]))
                ymin = int(np.min(pts[:, 1]))
                ymax = int(np.max(pts[:, 1]))

                logger.info(f"Detected text ('{text}' - confidence {prob:.2f}) at [{ymin}:{ymax}, {xmin}:{xmax}]")
                # if text is R or L, we can skip it as it's likely a label, but for now we will keep all detected text
                if text.strip().upper() in ['R', 'L']:
                    logger.info(f"Skipping detected text '{text}' as it is likely a left/right marker.")
                    continue
                if prob < 0.1:
                    logger.info(f"Skipping detected text '{text}' due to low confidence ({prob:.2f}).")
                    continue
                bboxes.append((ymin, xmin, ymax, xmax))
        else:
            gray = cv2.cvtColor(ocr_input, cv2.COLOR_RGB2GRAY) if ocr_input.ndim == 3 else norm_img
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if aspect_ratio > 1.5 and w > 20 and h > 8:
                    bboxes.append((y, x, y + h, x + w))

        return bboxes


class DICOMAnonymizer:
    """Handles parsing, text detection, pixel zeroing, DICOM export, and PNG comparison generation."""

    def __init__(self, detector: TextAnnotationDetector, padding: int = 5):
        self.detector = detector
        self.padding = padding

    @staticmethod
    def is_dicom_file(file_path: Path) -> bool:
        """Step 2: Check if file is a valid DICOM file."""
        if not file_path.is_file():
            return False
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except (InvalidDicomError, Exception):
            return False

    def _save_comparison_png(self, original_frame: np.ndarray, redacted_frame: np.ndarray, save_path: Path):
        """Generates and saves a side-by-side PNG comparison of original vs masked frame."""
        orig_norm = self.detector._normalize_to_uint8(original_frame)
        red_norm = self.detector._normalize_to_uint8(redacted_frame)

        if orig_norm.ndim == 2:
            orig_bgr = cv2.cvtColor(orig_norm, cv2.COLOR_GRAY2BGR)
            red_bgr = cv2.cvtColor(red_norm, cv2.COLOR_GRAY2BGR)
        elif orig_norm.ndim == 3 and orig_norm.shape[2] == 3:
            orig_bgr = cv2.cvtColor(orig_norm, cv2.COLOR_RGB2BGR)
            red_bgr = cv2.cvtColor(red_norm, cv2.COLOR_RGB2BGR)
        else:
            orig_bgr = orig_norm
            red_bgr = red_norm

        h, w = orig_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 800.0)
        thickness = max(1, int(font_scale * 2))

        cv2.putText(orig_bgr, "ORIGINAL", (15, max(30, int(h * 0.05))), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        cv2.putText(red_bgr, "MASKED", (15, max(30, int(h * 0.05))), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        comparison = np.hstack((orig_bgr, red_bgr))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), comparison)
        logger.info(f"Saved side-by-side comparison image: {save_path}")

    def redact_pixel_array(self, pixel_array: np.ndarray, samples_per_pixel: int = 1) -> Tuple[np.ndarray, bool]:
        """Performs object detection and fills detected text regions with zeros."""
        modified = False
        redacted_array = pixel_array.copy()

        if redacted_array.ndim == 2:
            frames = [redacted_array]
        elif redacted_array.ndim == 3 and samples_per_pixel == 3:
            frames = [redacted_array]
        elif redacted_array.ndim == 3 and samples_per_pixel == 1:
            frames = [redacted_array[i] for i in range(redacted_array.shape[0])]
        elif redacted_array.ndim == 4:
            frames = [redacted_array[i] for i in range(redacted_array.shape[0])]
        else:
            frames = [redacted_array]

        ring_margin = max(1, self.padding)  # Thickness of the surrounding pixel ring for calculating average intensity to fill the detected text region
        
        for idx, frame in enumerate(frames):
            bboxes = self.detector.detect_text_boxes(frame)
            if bboxes:
                modified = True
                h, w = frame.shape[:2]
                for (ymin, xmin, ymax, xmax) in bboxes:
                    p_ymin = max(0, ymin - self.padding)
                    p_ymax = min(h, ymax + self.padding)
                    p_xmin = max(0, xmin - self.padding)
                    p_xmax = min(w, xmax + self.padding)

                    # Define outer neighborhood boundaries clamped to image dimensions
                    r_ymin = max(0, p_ymin - ring_margin)
                    r_ymax = min(h, p_ymax + ring_margin)
                    r_xmin = max(0, p_xmin - ring_margin)
                    r_xmax = min(w, p_xmax + ring_margin)


                    # Crop the outer region and mask out the inner box to isolate the ring
                    outer_region = frame[r_ymin:r_ymax, r_xmin:r_xmax]
                    mask = np.ones(outer_region.shape[:2], dtype=bool)
                    mask[p_ymin - r_ymin : p_ymax - r_ymin, p_xmin - r_xmin : p_xmax - r_xmin] = False

                    # Compute mean color/intensity of surrounding ring pixels
                    surrounding_pixels = outer_region[mask]
                    if surrounding_pixels.size > 0:
                        avg_value = surrounding_pixels.mean(axis=0).astype(frame.dtype)
                    else:
                        avg_value = 0  # Fallback if the box occupies the entire frame
                                    
                
                    if frame.ndim == 2:
                        frame[p_ymin:p_ymax, p_xmin:p_xmax] = avg_value
                    else:
                        frame[p_ymin:p_ymax, p_xmin:p_xmax, :] = avg_value

                if redacted_array.ndim == 3 and samples_per_pixel == 1:
                    redacted_array[idx] = frame
                elif redacted_array.ndim == 4:
                    redacted_array[idx] = frame

        return redacted_array, modified

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Loads DICOM, detects text annotations, redacts, exports DICOM, and outputs PNG comparison."""
        
        logger.info(f"Processing DICOM file: {input_path}")
        
        try:
            ds = pydicom.dcmread(input_path)
        except Exception as e:
            logger.error(f"Failed to read DICOM file {input_path}: {e}")
            return False

        if 'PixelData' not in ds:
            logger.warning(f"No pixel data found in {input_path}. Copying file unchanged.")
            ds.save_as(output_path)
            return True

        
        img_array = ds.pixel_array.copy()
        # Summarize img_array properties for debugging
        logger.debug(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")
        
        
        samples_per_pixel = getattr(ds, 'SamplesPerPixel', 1)

        # Read raw buffer manually if .pixel_array still complains
        #dtype = np.uint16 if getattr(ds, "BitsAllocated", 16) == 16 else np.uint8
        #img_array = np.frombuffer(ds.PixelData, dtype=dtype)

        # Reshape based on DICOM dimensions
        #rows, cols = ds.Rows, ds.Columns
        #frames = getattr(ds, "NumberOfFrames", 1)

        # if frames > 1:
        #     img_array = img_array.reshape((frames, rows, cols))
        # else:
        #     img_array = img_array.reshape((rows, cols))
            

        modified_pixels, text_found = self.redact_pixel_array(img_array, samples_per_pixel)

        if text_found:
            logger.info(f"Text detected in {input_path.name}. Redacting pixel array...")
            
            if ds.file_meta.TransferSyntaxUID.is_compressed:
                # Wrap the raw compressed bytes in standard DICOM encapsulation tags
               # ds.PixelData = pydicom.encaps.encapsulate([ds.PixelData])  
                #ds.decompress()
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            
            ds.PixelData = modified_pixels.tobytes()

            

            if 'BurnedInAnnotation' in ds:
                ds.BurnedInAnnotation = 'NO'

            png_path = output_path.with_suffix('.png')
            if img_array.ndim == 2 or (img_array.ndim == 3 and samples_per_pixel == 3):
                self._save_comparison_png(img_array, modified_pixels, png_path)
            else:
                self._save_comparison_png(img_array[0], modified_pixels[0], png_path)
        else:
            logger.info(f"No text detected in {input_path.name}.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(output_path)
        logger.info(f"Saved processed DICOM to: {output_path}")
        return True

    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Loops through directory and processes all DICOM files."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        total_files = 0
        dicom_files = 0
        anonymized_files = 0

        logger.info(f"Scanning directory: {input_dir}")

        for root, _, files in os.walk(input_dir):
            for file_name in files:
                total_files += 1
                file_path = Path(root) / file_name

                if not self.is_dicom_file(file_path):
                    logger.debug(f"Skipping non-DICOM file: {file_path}")
                    continue

                dicom_files += 1
                rel_path = file_path.relative_to(input_dir)
                dest_path = output_dir / rel_path

                if self.process_file(file_path, dest_path):
                    anonymized_files += 1

        logger.info("=" * 50)
        logger.info(f"Processing Complete:")
        logger.info(f"  Total files scanned: {total_files}")
        logger.info(f"  Valid DICOM files:   {dicom_files}")
        logger.info(f"  Successfully saved:  {anonymized_files}")
        logger.info("=" * 50)


if __name__ == "__main__":
    parser = CustomArgumentParser(
        description="DICOM Image Text Annotation Anonymizer with PNG Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python dicom_anonymizer.py -i ./input_dicoms -o ./output_dicoms
  python dicom_anonymizer.py -i ./input_dicoms -o ./output_dicoms --gpu --padding 10
"""
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input directory containing DICOM files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for anonymized files")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for OCR engine")
    parser.add_argument("--padding", type=int, default=5, help="Pixel padding around detected text bounding box (default: 5)")

    # Print usage/help if executed without any CLI arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    detector_engine = TextAnnotationDetector(use_gpu=args.gpu)
    anonymizer = DICOMAnonymizer(detector=detector_engine, padding=args.padding)
    anonymizer.process_directory(args.input, args.output)