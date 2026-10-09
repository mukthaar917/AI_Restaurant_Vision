import os
import subprocess
import imageio_ffmpeg

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
output_dir = "output_videos"
web_dir = os.path.join(output_dir, "web_ready")
os.makedirs(web_dir, exist_ok=True)

for fname in os.listdir(output_dir):
    if fname.endswith(".mp4") and not fname.startswith("Web_"):
        in_file = os.path.join(output_dir, fname)
        out_file = os.path.join(web_dir, f"Web_{fname}")
        
        print(f"Converting {fname} for web/cloud upload...")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", in_file,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "fast",
            "-movflags", "+faststart",
            out_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved: {out_file}")

print("\nDone! Upload the files located in 'output_videos/web_ready/'.")