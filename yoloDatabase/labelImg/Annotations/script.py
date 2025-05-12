import os
import argparse

def rename_files(folder_path, offset):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            basename, ext = os.path.splitext(filename)
            if basename.isdigit():
                num = int(basename)
                new_num = num + offset
                new_filename = f"{new_num}{ext}"
                new_filepath = os.path.join(folder_path, new_filename)
                if not os.path.exists(new_filepath):
                    os.rename(filepath, new_filepath)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                else:
                    print(f"Error: '{new_filename}' already exists. Skipping '{filename}'.")
            else:
                print(f"Skipping '{filename}' (not a numeric name)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename numeric files by adding a fixed offset.")
    parser.add_argument("folder", help="Path to the target folder containing files")
    parser.add_argument("--offset", type=int, default=1000, help="Fixed offset to add (default: 1000)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory.")
        exit(1)
    
    rename_files(args.folder, args.offset)
