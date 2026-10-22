# remove_burnedin_annotations
Code to support anonymization of DICOM files with burned in text annotations in image

See head of `remove_burnedin_annotations.py` for install and usage instructions

## Summary

remove_burnedin_annotations.py

Steps:
1) Loops through a directory of dicom files
2) Load DICOM file with pydicom
3) Performs Optical Character Recognition to detect text in the image
4) If not a "L" or "R" marker, masks out region of text
5) If text detected, save the image as a DICOM file with the same header as the input
6) Saves a *.png file with the before and after masked image


## Usage

Examples:
  
  python remove_burnedin_annotations.py -i input_dicom_dir -o output_dicom_dir --gpu --padding 5

