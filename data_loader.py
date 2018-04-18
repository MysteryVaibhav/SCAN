import torch.utils.data
from util import *


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, ids, regions_in_image, visual_feature_dimension, image_features_dir):
        self.img_one_hot = img_one_hot
        self.ids = ids
        self.num_of_samples = len(ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        input, mask = self.img_one_hot[self.ids[idx]]

        #image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        image = np.load(self.image_features_dir + "{}.npy".format(self.ids[idx].split("#")[0])).reshape(
                       (self.regions_in_image, self.visual_feature_dimension))

        r_n = idx
        img_idx = self.ids[idx].split("#")[0]
        r_n_idx = img_idx
        while r_n_idx == img_idx:
            r_n = np.random.randint(self.num_of_samples)
            r_n_idx = self.ids[r_n].split("#")[0]

        # Return negative caption and image
        image_neg = np.load(self.image_features_dir + "{}.npy".format(self.ids[r_n].split("#")[0])).reshape(
                           (self.regions_in_image, self.visual_feature_dimension))
        #image_neg = np.random.random((self.regions_in_image, self.visual_feature_dimension))

        input_neg, mask_neg = self.img_one_hot[self.ids[r_n]]

        return to_tensor(input).long(), to_tensor(mask), to_tensor(image), \
               to_tensor(input_neg).long(), to_tensor(mask_neg), to_tensor(image_neg)


def get_k_random_numbers(n, curr, k=16):
    random_indices = set()
    while len(random_indices) < k:
        idx = np.random.randint(n)
        if idx != curr and idx not in random_indices:
            random_indices.add(idx)
    return list(random_indices)


class CustomDataSet1(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, caption_ids, image_ids, regions_in_image, visual_feature_dimension, image_features_dir):
        self.img_one_hot = img_one_hot
        self.caption_ids = caption_ids
        self.image_ids = image_ids
        self.num_of_samples = len(self.caption_ids)
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir
        self.all_image_features = self.get_all_image_features()

    def __len__(self):
        return self.num_of_samples

    def get_all_image_features(self):
        image_features = np.zeros((len(self.image_ids), self.regions_in_image, self.visual_feature_dimension))
        # Get all the 1000 image features
        for i, id in enumerate(self.image_ids):
            #image_features[i] = np.random.random((self.regions_in_image, self.visual_feature_dimension))
            image_features[i] = np.load(self.image_features_dir + "{}.npy".format(id)).reshape(
                                       (self.regions_in_image, self.visual_feature_dimension))
        return image_features

    def __getitem__(self, idx):
        # Get the caption and mask
        caption_one_hot, caption_mask = self.img_one_hot[self.caption_ids[idx]]
        return to_tensor(caption_one_hot).long(), to_tensor(caption_mask), to_tensor(self.all_image_features), self.caption_ids[idx].split("#")[0]


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.img_one_hot = run(params.caption_file)
        self.train_ids = get_ids('train', params.split_file)
        self.val_ids = get_ids('val', params.split_file)
        self.plain_val_ids = get_ids('val', params.split_file, strip=True)
        self.test_ids = get_ids('test', params.split_file)
        self.plain_test_ids = get_ids('test', params.split_file, strip=True)
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        #kwargs = {} if torch.cuda.is_available() else {}
        self.training_data_loader = torch.utils.data.DataLoader(CustomDataSet(self.img_one_hot,
                                                                              self.train_ids,
                                                                              params.regions_in_image,
                                                                              params.visual_feature_dimension,
                                                                              params.image_features_dir
                                                                              ),
                                                                batch_size=self.params.batch_size,
                                                                shuffle=True, **kwargs)
        self.eval_data_loader = torch.utils.data.DataLoader(CustomDataSet1(self.img_one_hot,
                                                                           self.val_ids,
                                                                           self.plain_val_ids,
                                                                           params.regions_in_image,
                                                                           params.visual_feature_dimension,
                                                                           params.image_features_dir),
                                                            batch_size=1, shuffle=False, **kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(CustomDataSet1(self.img_one_hot,
                                                                           self.test_ids,
                                                                           self.plain_test_ids,
                                                                           params.regions_in_image,
                                                                           params.visual_feature_dimension,
                                                                           params.image_features_dir),
                                                            batch_size=1, shuffle=False, **kwargs)

    @staticmethod
    def hard_negative_mining(model, pos_cap, pos_mask, pos_image, neg_cap, neg_mask, neg_image):
        model.eval()
        bs = len(pos_image)

        hard_neg_cap = torch.LongTensor(neg_cap.size())
        hard_neg_mask = torch.FloatTensor(neg_mask.size())
        hard_neg_img = torch.FloatTensor(neg_image.size())
        for i in range(bs):

            each_image = pos_image[i].unsqueeze(0).repeat(bs, 1, 1)
            similarity = model(to_variable(neg_cap), to_variable(neg_mask), to_variable(each_image), False).data.cpu().numpy()
            hardest_neg = np.argmax(similarity)
            hard_neg_cap[i] = neg_cap[hardest_neg]
            hard_neg_mask[i] = neg_mask[hardest_neg]

            each_cap = pos_cap[i].unsqueeze(0).repeat(bs, 1)
            each_mask = pos_mask[i].unsqueeze(0).repeat(bs, 1)
            similarity = model(to_variable(each_cap), to_variable(each_mask), to_variable(neg_image), False).data.cpu().numpy()
            hardest_neg = np.argmax(similarity)
            hard_neg_img[i] = neg_image[hardest_neg]

        return pos_cap, pos_mask, pos_image, hard_neg_cap, hard_neg_mask, hard_neg_img
