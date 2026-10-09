Set-Content -Path README.md -Value @"

AI Restaurant Vision \& Compliance Intelligence System


An end-to-end computer vision and spatial analytics pipeline designed for restaurant monitoring, food safety compliance, and perimeter security across multi-camera CCTV streams.


Key Features \& Use Cases



1. Front Gate Patron Counting \& Flow Tracking (Camera 2)

&#x20;  ROI Gateway Tracking\*\*: Monitors entrance threshold crossings with persistent IDs.

&#x20;  Occupancy Counter\*\*: Dynamically toggles between \\`Inside : 0\\` and \\`Inside : 1\\` upon patron entry and exit.



2. Mandatory Employee PPE Compliance (Cameras 3, 4, 5, 6)

   Haircap Detection: Real-time auditing of worn hairnets and hygiene caps.

   Apron Auditing: Adaptive luminance and coverage tracking for disposable kitchen aprons.

   Transparent \& Hygiene Glove Tracking: Uses anatomical wrist and hand keypoints (\\`YOLOv8-Pose\\`) to verify glove compliance,                suppressing phantom bounding boxes on clothing or back panels.

   Vacant Zone Handling: Automatically suppresses static employee labels when prep stations are empty.



3. Workplace Device Usage Detection (Camera 4)

   Active Handheld Device Monitoring\*\*: Detects mobile phone usage in food-prep zones and flags \\`ALARM: PHONE IN USE\\`.



4. Restricted Zone Access Tripwires (Camera 6)

    Storeroom Access Alerting\*\*: Instant intrusion detection triggers \\`ALARM: STOREROOM ACCESS\\` when personnel enter inventory areas.



5. Production Visuals \& Video Encoding

   Clean Drop-Shadow HUD\*\*: Renders transparent overlays without opaque black backgrounds.

   Universal H.264 MP4 Export\*\*: Automatically encodes processed streams with faststart flags for browser and cloud playback.


Installation \& Setup



\\`\\`\\`bash

git clone https://github.com/mukthaar917/AI\_Restaurant\_Vision.git

cd AI\_Restaurant\_Vision

python -m venv .venv

.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

\\`\\`\\`


Run Pipeline


\\`\\`\\`bash

python run\_demo.py

\\`\\`\\`

"@

