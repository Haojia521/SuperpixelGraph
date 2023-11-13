from .WHU import DatasetWhu


def get_dataset_trainval(name, root, transform=None, transform_val=None):
    if name == 'WHU':
        train_dataset = DatasetWhu(root, 'train', transform)
        val_dataset = DatasetWhu(root, 'val', transform_val)
    else:
        raise ValueError(f'unknown dataset: {name}')

    return train_dataset, val_dataset


def get_dataset_test(name, root, transform=None):
    if name == 'WHU':
        test_dataset = DatasetWhu(root, 'test', transform)
    else:
        raise ValueError(f'unknown dataset: {name}')

    return test_dataset
