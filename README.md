# Reactor Power Monitoring using Cherenkov Radiation with a Commercial Camera System
This repository contains the source code and data for the research paper:
"Reactor Power Monitoring using Cherenkov Radiation with a Commercial Camera System"

---

### Prerequisites
To process the raw video data, you must have FFmpeg installed on your system.
* FFmpeg: https://ffmpeg.org/download.html

### Install Python Dependencies
All required Python packages can be installed using the provided `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Data Acquisition
Due to the large file size, the raw video datasets used in this study are hosted externally.

* Download Link: https://utexas.box.com/s/bd80q5yukeabywpuq04h31nhen6z6tbj

---

## Data Processing
After downloading the videos, place them in the appropriate directory and run:

```bash
python 0_preprocessing.py
```

This script performs:
- Cropping and concatenation of video segments
- Denoising using FFmpeg
- Extraction of per-frame RGB intensity values

---

## Additional Resources
Related work and project context can be found on the Digital Twin website developed under the
**Digital Molten Salt Reactor Initiative**, led by **Dr. Kevin T. Clarno (UT Austin)**:

https://nuclear-twins.tacc.utexas.edu/cherenkov