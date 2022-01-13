import os

for root, dirs, files in os.walk("training/source/", topdown=False):
    for file in files:
        # Ignore files that we cant parse
        if not file.endswith(".rle"):
            continue

        data = ""
        file_path = root + file
        with open(file_path, "r") as f:
            data = f.read()

        file_path = "training/positive/" if "oscillator" in data else "training/negative/"
        file_path += file

        with open(file_path, "w") as f:
            f.write(data)
