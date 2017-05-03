def ensure_dir(filepath):
    import os
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return filepath


def main():
    return


if __name__ == "__main__":
    main()
