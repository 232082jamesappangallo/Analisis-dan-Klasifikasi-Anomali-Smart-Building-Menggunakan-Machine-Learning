import json

# Load notebook
with open('scripts/ML_Pipeline_Notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cells
for cell in nb['cells']:
    # Fix source - should be list of strings
    if isinstance(cell['source'], str):
        lines = cell['source'].split('\n')
        cell['source'] = lines
    
    # Remove invalid id field
    if 'id' in cell:
        del cell['id']

# Save fixed notebook
with open('scripts/ML_Pipeline_Notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✅ Notebook fixed")
