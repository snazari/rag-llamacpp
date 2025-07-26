import os
import shutil
import argparse
from tqdm import tqdm

# --- Configuration ---
# Maps file extensions to their destination subfolder.
FILE_TYPE_MAPPING = {
    '.md': 'md',
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.pptx': 'pptx',
    '.txt': 'txt',
}

def handle_collision(dest_path):
    """Generates a new path if a file already exists to avoid overwriting.

    Args:
        dest_path (str): The original desired destination path.

    Returns:
        str: A new, unique destination path.
    """
    base, ext = os.path.splitext(dest_path)
    counter = 1
    new_dest_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_dest_path):
        counter += 1
        new_dest_path = f"{base}_{counter}{ext}"
    return new_dest_path

def organize_files(source_dir, dest_dir):
    """Recursively finds, copies, and organizes files into subdirectories.

    Args:
        source_dir (str): The directory to search recursively.
        dest_dir (str): The root directory to place organized files.
    """
    print(f"Scanning '{source_dir}' for documents...")

    # Create all necessary destination subdirectories first.
    for subdir in set(FILE_TYPE_MAPPING.values()):
        os.makedirs(os.path.join(dest_dir, subdir), exist_ok=True)

    # First, collect a list of all files to be copied.
    files_to_copy = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Get the file extension and convert to lower case for matching.
            ext = os.path.splitext(file)[1].lower()
            if ext in FILE_TYPE_MAPPING:
                files_to_copy.append((os.path.join(root, file), ext))

    if not files_to_copy:
        print("No supported documents found.")
        return

    print(f"Found {len(files_to_copy)} supported documents. Starting organization...")

    # Now, copy the files with a progress bar.
    for source_path, ext in tqdm(files_to_copy, desc="Organizing files"):
        try:
            subdir = FILE_TYPE_MAPPING[ext]
            dest_path = os.path.join(dest_dir, subdir, os.path.basename(source_path))

            # If a file with the same name exists, find a new name.
            if os.path.exists(dest_path):
                dest_path = handle_collision(dest_path)
            
            shutil.copy(source_path, dest_path)
        except Exception as e:
            print(f"\nError copying file {source_path}: {e}")

def main():
    """Main function to parse arguments and run the organizer."""
    parser = argparse.ArgumentParser(
        description="Recursively find and organize documents from a source directory into a structured destination directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_directory", 
        help="The directory to search for documents."
    )
    parser.add_argument(
        "destination_directory", 
        help="The root directory where organized files will be placed (e.g., './organized_docs')."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_directory):
        print(f"Error: Source directory not found at '{args.source_directory}'")
        return

    organize_files(args.source_directory, args.destination_directory)
    print(f"\nFile organization complete.")
    print(f"All documents have been copied to subfolders within: '{args.destination_directory}'")

if __name__ == "__main__":
    main()
