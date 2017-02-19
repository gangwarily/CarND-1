import LaneLinesMarker as marker
from moviepy.editor import VideoFileClip

#
# This class is for processing videos using the pipeline built in LaneLinesMarker.
#

# Method for processing video clip.
def process_video(output_name, file_name):
    video_output = output_name
    clip1 = VideoFileClip(file_name)
    video_clip = clip1.fl_image(marker.process_image)
    video_clip.write_videofile(video_output, audio=False)

# MAIN METHOD
process_video("white_output.mp4", "solidWhiteRight.mp4")
process_video("yellow_output.mp4", "solidYellowLeft.mp4")
# process_video("extra.mp4", "challenge.mp4") # challenge video
