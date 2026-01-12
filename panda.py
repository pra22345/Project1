import panda as pd
import os

file_path = r"E:\practice\data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
else:
    print(f"Error: File not found at {file_path}")