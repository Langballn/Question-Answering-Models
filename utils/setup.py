import os
from six.moves.urllib.request import urlretrieve
import zipfile


def download_embeddings(download_url, filename, target_dir):
    '''
    Downloads and unzips a file from a specifed URL to a specifed directory
    '''

    # check that the target directory exists, if not create it
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    url = os.path.join(download_url, filename)
    file_path = os.path.join(target_dir, filename)

    # download data fron the URL specifed
    print('Downloading data from', url, '...')
    try:
        urlretrieve(url, file_path)
    except:
        print('Error downloading data from ', url)

    # unzip files saving result to the target directory
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
    except:
        print('Error unzipping data from ', file_path)

    print('Data has been successfully saved in ', target_dir)
