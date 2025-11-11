import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog

# Hide the main tkinter window
root = Tk()
root.withdraw()

# Open file dialog to select Excel file
print("Please select your Excel file...")
file_path = filedialog.askopenfilename(
    title="Select Excel File",
    filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected. Exiting...")
    exit()

print(f"Loading file: {file_path}")

# Read the Excel file
df = pd.read_excel(file_path)

# Display available columns
print("\nAvailable columns in the file:")
print(df.columns.tolist())

# Extract NDTI columns (adjust column names if needed)
ndti_columns = ['NDTI P50', 'NDTI p0', 'NDTI p100']

# Check if columns exist and get actual column names (case-insensitive)
actual_columns = []
for col in ndti_columns:
    found = False
    for df_col in df.columns:
        if col.lower() == df_col.lower():
            actual_columns.append(df_col)
            found = True
            break
    if not found:
        print(f"Warning: Column '{col}' not found in the file")

if not actual_columns:
    print("\nError: No NDTI columns found. Please check your column names.")
    exit()

# Create figure with subplots
fig, axes = plt.subplots(len(actual_columns), 1, figsize=(10, 5 * len(actual_columns)))

# Handle case where there's only one column (axes won't be an array)
if len(actual_columns) == 1:
    axes = [axes]

# Create histogram for each column
for idx, col in enumerate(actual_columns):
    # Remove NaN values
    data = df[col].dropna()
    
    if len(data) == 0:
        print(f"Warning: No valid data in column '{col}'")
        continue
    
    # Calculate statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    
    # Determine optimal number of bins using Sturges' rule
    n_bins = int(np.ceil(np.log2(len(data)) + 1))
    n_bins = max(10, min(n_bins, 50))  # Keep between 10 and 50 bins
    
    # Create histogram
    counts, bins, patches = axes[idx].hist(data, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Find the dominant bin
    max_count_idx = np.argmax(counts)
    patches[max_count_idx].set_facecolor('darkred')
    patches[max_count_idx].set_alpha(1.0)
    
    # Add vertical lines for mean and median
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    # Labels and title
    axes[idx].set_xlabel('Value', fontsize=12)
    axes[idx].set_ylabel('Frequency', fontsize=12)
    axes[idx].set_title(f'Histogram of {col}\n(n={len(data)}, std={std_val:.2f})', fontsize=14, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    
    # Print dominant range
    dominant_range = f"[{bins[max_count_idx]:.2f}, {bins[max_count_idx + 1]:.2f})"
    print(f"\n{col}:")
    print(f"  Dominant range: {dominant_range} with {int(counts[max_count_idx])} data points")
    print(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}, Std Dev: {std_val:.2f}")

plt.tight_layout()
plt.show()

print("\nHistogram generation complete!")