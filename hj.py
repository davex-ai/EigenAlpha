import os


def combine_py_files(root_dir, output_file, skip_dirs=None, skip_files=None):
    """
    Reads only .py files and adds their file name/path to the top of their content.
    """
    skip_dirs = skip_dirs or []
    skip_files = skip_files or []

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            # Exclude specific subfolders from the walk
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for filename in files:
                # Filter for .py files only
                if not filename.endswith('.py'):
                    continue

                # Skip explicit files or the output file itself
                if filename in skip_files or filename == output_file:
                    continue

                file_path = os.path.join(root, filename)

                # HEADER: Writing file name and path at the top of the code
                outfile.write(f"\n{'#' * 80}\n")
                outfile.write(f"# FILE NAME: {filename}\n")
                outfile.write(f"# FULL PATH: {file_path}\n")
                outfile.write(f"{'#' * 80}\n\n")

                try:
                    # Open in read-only mode ('r')
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"# [Error reading file: {e}]\n")

                # Add a few line breaks between files for readability
                outfile.write("\n\n")


if __name__ == "__main__":
    # --- Configuration ---
    # Using 'r' for raw string to handle Windows backslashes correctly
    target_path = r"C:\Users\DELL\PycharmProjects\EigenAlpha"
    result_file = "all_py_code_combined.txt"

    # Folders and Files to ignore
    exclude_folders = ['.git', 'node_modules', '__pycache__', '.venv', 'backend-stg-0']
    exclude_filenames = ['hj.py', 'data.py', '.gitignore', 'secrets.env', 'config.json', 'sp500_tickers.csv']

    print(f"Reading .py files in {os.path.abspath(target_path)}...")
    combine_py_files(target_path, result_file, exclude_folders, exclude_filenames)
    print(f"Success! All Python code combined into: {result_file}")
