import os
import subprocess

def zip_directories_with_prefix(base_path, prefix):
    for item in os.listdir(base_path):
        dir_path = os.path.join(base_path, item)

        # Check if it's a directory with the given prefix
        if os.path.isdir(dir_path) and item.startswith(prefix):
            zip_path = os.path.join(base_path, f"{item}.zip")

            # Check if zip already exists
            if os.path.exists(zip_path):
                print(f"Skipping (already zipped): {item}")
                continue

            print(f"Zipping in background: {item}")

            # Run zip command in background, no output
            cmd = f"nohup zip -r {base_path}{item}.zip {base_path}{item} > /dev/null 2>&1 &"
            # print(f"Executing command: {cmd}")
            p = subprocess.Popen(cmd, shell=True)
            print(f"Started zipping {item} with PID: {p.pid}")

    print("All jobs started.")

# Example usage
zip_directories_with_prefix(base_path="../", prefix="Results_")