import yt_dlp
import boto3
import os
import tempfile
from flask import jsonify
from dotenv import load_dotenv




load_dotenv()

def save_audio(URLS, s3_bucket, s3_key_prefix=""):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    try:
        # Create a temporary directory; files here will be cleaned up once done.
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Set yt-dlp options to store the file in the temporary directory
            ydl_opts = {
                'format': 'm4a/bestaudio/best',
                'outtmpl': os.path.join(tmpdirname, '%(id)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                }]
            }

            # Download the file using yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(URLS)

            # After download, iterate over the temporary directory to upload files.
            for filename in os.listdir(tmpdirname):
                filepath = os.path.join(tmpdirname, filename)
                # Build the S3 key: include prefix if provided
                s3_key = os.path.join(s3_key_prefix, filename) if s3_key_prefix else filename

                # Upload file to S3
                s3_client.upload_file(filepath, s3_bucket, s3_key)
                print(f"Uploaded {filename} to s3://{s3_bucket}/{s3_key}")
                break

        # Return a successful JSON response
        return jsonify({"message": "Audio downloaded and uploaded successfully to S3.", "key": s3_key}), 200

    except Exception as e:
        # Return error JSON response
        return jsonify({"error": str(e)}), 500
    