import tarfile
import os
import numpy as np

def load_cifar10_images(path):

    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffer = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffer[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Labels are the first byte of every chunk
    labels = buffer[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffer, np.arange(0, buffer.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]


def calculate_euclidean_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))


def get_euclidean_distance_between_images(train_images, test_images):
    f = open("/Users/user/PycharmProjects/result.txt", "x")
    for x in np.nditer(train_images):
        for y in np.nditer(test_images):
            dist = calculate_euclidean_distance(x, y)
            f.write(str(dist))
            f.write("\n")
        f.write("...........")









