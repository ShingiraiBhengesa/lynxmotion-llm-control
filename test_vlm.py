from vision.camera import LogitechCamera
from vlm_controller import VLMController
from datetime import datetime

camera = LogitechCamera()
vlm = VLMController(debug=True)
ret, frame = camera.capture_frame()
if ret:
    objects = vlm.detect_objects(frame)
    for obj in objects:
        print(f"Detected: {obj['label']} at {obj['center_mm']} at {datetime.now().strftime('%I:%M %p %Z, %m/%d/%Y')}")
    command = "move to red object"
    vlm.process_command(command)
    print(f"Check debug_images/ for visualization at {datetime.now().strftime('%I:%M %p %Z, %m/%d/%Y')}")
else:
    print(f"❌ Failed to capture frame at {datetime.now().strftime('%I:%M %p %Z, %m/%d/%Y')}")
camera.release()