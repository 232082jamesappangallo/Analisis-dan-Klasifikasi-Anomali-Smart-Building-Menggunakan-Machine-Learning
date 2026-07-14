import os
import subprocess

# Change to project directory
os.chdir(r'c:\File D\project')

# Run notebook with execute
result = subprocess.run([
    'python', '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook',
    '--execute',
    'scripts/ML_Pipeline_Notebook.ipynb',
    '--inplace'
], capture_output=True, text=True, timeout=600)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)
