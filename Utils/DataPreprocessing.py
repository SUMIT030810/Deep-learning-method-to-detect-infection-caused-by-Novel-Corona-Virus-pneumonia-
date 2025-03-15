from torchvision.transforms import transforms


def get_train_transform(train_mean,train_std):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])
    print(f"Transform applied on train data...")
    return train_transform


def get_test_transform(test_mean,test_std):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=test_mean, std=test_std)
    ])
    print(f"Transform applied on test data...")
    return test_transform