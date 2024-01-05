import helper_functions as hf

videos_folder = "/home/andrei/Work/datasets/CVPR2023 Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results/test videos/test videos"
annotations_folder = "/home/andrei/Work/datasets/CVPR2023 Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results/test videos/annotations"
video_frames = "/home/andrei/Work/datasets/CVPR2023 Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results/test videos/videos_frames (full)"
annotated_videos = "/home/andrei/Work/datasets/CVPR2023 Rip Current Segmentation: A Novel Benchmark and YOLOv8 Baseline Results/test videos/annotated videos"

# display annotations over videos
hf.process_videos(videos_folder, annotations_folder, annotated_videos)
