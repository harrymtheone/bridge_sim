import time

import numpy as np
from vispy import app, scene


class ImageViewer:
    def __init__(self, shape=(512, 512, 3)):
        # Create a SceneCanvas (has a scene)
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.image = scene.visuals.Image(np.zeros(shape, dtype=np.uint8), parent=self.view.scene)
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1

    def update_image(self, img):
        """Call this to update the displayed image."""
        self.image.set_data(img)
        self.canvas.update()  # Force canvas update
        app.process_events()  # Process events to refresh the UI


# Usage example:
viewer = ImageViewer()

# Method 1: Use a timer for automatic updates
def update_timer(event):
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    viewer.update_image(img)
    print(f"Updated frame")

# Create timer that fires every 100ms
timer = app.Timer(0.1, connect=update_timer, start=True)

# Start the event loop
app.run()

# Alternative Method 2 (comment out the above and uncomment below):
# for i in range(300):
#     img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
#     viewer.update_image(img)
#     time.sleep(0.1)
#     print(i)
#     
#     # Check if window was closed
#     if not viewer.canvas.native:
#         break
#
# # Keep window open after updates finish
# if viewer.canvas.native:
#     app.run()
