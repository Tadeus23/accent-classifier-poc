import os
from app.config import logging

def cleanup_temp_files(*file_paths: str):
    """
    Remove temporary files.
    
    Args:
        *file_paths (str): Paths to files to be deleted.
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")