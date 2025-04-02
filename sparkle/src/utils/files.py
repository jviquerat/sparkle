import filecmp


# Compare files
def compare_files(f1, f2):

    return filecmp.cmp(f1, f2, shallow=False)
