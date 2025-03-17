# Generic imports
import warnings
import filecmp

# Filter warning messages
warnings.filterwarnings('ignore',category=DeprecationWarning)

# Compare files
def compare_files(f1, f2):

    return filecmp.cmp(f1, f2, shallow=False)
