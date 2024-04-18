import os

def get_file_size(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return size_in_bytes

def convert_size(size_in_bytes):
    for unit in ['bytes']:
        if size_in_bytes < 1024.0:
            break
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} {unit}"

file_path_video = "path/to/your/video.file"
file_path_image = "path/to/your/image.file"
file_path_audio = "path/to/your/audio.file"

video_size = get_file_size(file_path_video)
image_size = get_file_size(file_path_image)
audio_size = get_file_size(file_path_audio)

print(f"Video memory occupied: {convert_size(video_size)}")
print(f"Image memory occupied: {convert_size(image_size)}")
print(f"Audio memory occupied: {convert_size(audio_size)}")
