import filecmp


def compare_files(f1, f2):
    """
    Compares two files to check if they are identical.

    Args:
        f1: The path to the first file.
        f2: The path to the second file.

    Returns:
        True if the files are identical, False otherwise.
    """

    return filecmp.cmp(f1, f2, shallow=False)
