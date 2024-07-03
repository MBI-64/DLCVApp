# DLCVApp
A GUI application developed in Python to automate the process of handling multiple input sources for image and video data and applying various preprocessing to it. Solution to an assignment requiring the following features to be implemented:
1. Selection of input from a) image folder, b) offline video file c) Live web cam stream. File/folder selection dialog must be displayed for selection of image folder or video file. For live camstream, default USB camera may be selected.
2. Selection of playback speed from a frame rate menu.
3. Option to start/stop displaying frames in a continuous mode (restart from frame-1 again after last frame)
4. Main GUI should not hang while playing the frame. You can achieve this by using a separate video reading thread.
5. Bonus feature (optional): You can also add a new menu named filter and add options for various filters such as grayscale video, edge filter to highlight edges in the frames while playing etc.
