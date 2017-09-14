import os

def create_dir_if_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == '__main__':
    create_dir_if_exists('data')
    create_dir_if_exists('output')