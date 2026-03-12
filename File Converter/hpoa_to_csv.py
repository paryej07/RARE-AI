# convert_hpoa.py
import pandas as pd

filepath = "/Users/paryejalam/Downloads/phenotype.hpoa"

# Find the actual header line (first line that doesn't start with #)
header_line = 0
with open(filepath, "r") as f:
    for i, line in enumerate(f):
        if not line.startswith("#"):
            header_line = i
            break

print(f"Header found at line: {header_line}")

# Read the file skipping all comment lines
df = pd.read_csv(
    filepath,
    sep="\t",
    skiprows=header_line,
    header=0,
    on_bad_lines="skip"   # skip any malformed lines
)

print("Columns:", df.columns.tolist())
print(f"Rows: {len(df)}")
print(df.head())

# Save to CSV
output_path = "/Users/paryejalam/Downloads/hpoa_phenotypes.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")