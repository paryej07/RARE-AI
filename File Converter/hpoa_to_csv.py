# convert_hpoa.py
import pandas as pd

source_path = "/Users/paryejalam/Downloads/phenotype.hpoa"

# Find the actual header line (first line that doesn't start with #)
start_line = 0
with open(source_path, "r") as file:
    for idx, line in enumerate(file):
        if not line.startswith("#"):
            start_line = idx
            break

print(f"Header found at line: {start_line}")

# Read the file skipping all comment lines
data_frame = pd.read_csv(
    source_path,
    sep="\t",
    skiprows=start_line,
    header=0,
    on_bad_lines="skip"   # skip any malformed lines
)

print("Columns:", data_frame.columns.tolist())
print(f"Rows: {len(data_frame)}")
print(data_frame.head())

# Save to CSV
dest_path = "/Users/paryejalam/Downloads/hpoa_phenotypes.csv"
data_frame.to_csv(dest_path, index=False)
print(f"\nSaved to {dest_path}")
