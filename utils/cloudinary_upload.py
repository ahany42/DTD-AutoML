import os
import uuid
import cloudinary
import cloudinary.uploader

def upload_to_cloudinary(file_obj_or_path, folder="gp_output"):
    """
    Uploads a file or file-like object to Cloudinary.
    
    Args:
        file_obj_or_path: The file path (str) or a file-like object (e.g., BytesIO) containing the image.
        folder: The folder in Cloudinary where the image will be stored.
        
    Returns:
        str: The secure URL of the uploaded image on Cloudinary, or None if upload fails.
    """
    try:
        response = cloudinary.uploader.upload(
            file_obj_or_path,
            folder=folder,
            public_id=str(uuid.uuid4())
        )
        return response.get('secure_url')
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None
