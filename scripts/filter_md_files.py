import os
import shutil
import argparse


def filter_and_copy_markdown_files(input_folder, output_folder):
    """
    Traverses a folder and its subfolders to copy all .md files while preserving the folder structure.
    :param input_folder: The folder containing the files.
    :param output_folder: The folder where .md files will be copied with the same structure.
    """
    try:
        # Checks if the source folder exists
        if not os.path.isdir(input_folder):
            print(f"The folder '{input_folder}' does not exist.")
            return

        md_files_count = 0

        # Recursively traverses the folder
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.md'):
                    source_path = os.path.join(root, file)

                    # Recreates the relative folder structure
                    relative_path = os.path.relpath(root, input_folder)
                    destination_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(destination_dir, exist_ok=True)

                    destination_path = os.path.join(destination_dir, file)

                    # Checks if a file with the same name already exists and avoids overwriting
                    counter = 1
                    while os.path.exists(destination_path):
                        name, ext = os.path.splitext(file)
                        destination_path = os.path.join(destination_dir, f"{name}_{counter}{ext}")
                        counter += 1

                    shutil.copy2(source_path, destination_path)
                    md_files_count += 1

        print(f"{md_files_count}.md files copied to '{output_folder}', preserving the folder structure.")

    except Exception as e:
        print(f"Error: {e}")


# Path definitions
if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Copy Markdown files from source to destination, preserving folder structure.")
    parser.add_argument("-s", "--source", help="Source directory containing Markdown files")
    parser.add_argument("-d", "--dest", help="Destination directory for copied Markdown files")

    args = parser.parse_args()

    # If arguments are provided, use them
    if args.source and args.dest:
        dossier_source = args.source
        dossier_sortie = args.dest
    else:
        # Otherwise, prompt the user
        print("Please enter the following paths:")
        dossier_source = input("Source directory: ")
        dossier_sortie = input("Destination directory: ")

    filter_and_copy_markdown_files(dossier_source, dossier_sortie)
