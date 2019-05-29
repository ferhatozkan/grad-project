def read_samples2(folder_path):
    images = [Image.open(file, 'r') for file in glob.glob(folder_path + '*png')]
    return images