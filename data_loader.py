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


class CustomDataSet1(torch.utils.data.TensorDataset):
    def __init__(self, img_one_hot, caption_ids, image_ids, regions_in_image, visual_feature_dimension, image_features_dir, max_caption_len):
        self.img_one_hot = img_one_hot
        self.caption_ids = caption_ids
        self.image_ids = image_ids
        self.num_of_samples = len(self.image_ids)
        self.max_caption_len = max_caption_len
        self.regions_in_image = regions_in_image
        self.visual_feature_dimension = visual_feature_dimension
        self.image_features_dir = image_features_dir
        self.all_text_features, self.all_text_features_mask = self.get_all_text_features()

    def __len__(self):
        return self.num_of_samples

    def get_all_text_features(self):
        text_features = np.zeros((len(self.caption_ids), self.max_caption_len))
        text_features_mask = np.zeros((len(self.caption_ids), self.max_caption_len))
        # Get all the 5000 text features
        for i, id in enumerate(self.caption_ids):
            text_features[i], text_features_mask[i] = self.img_one_hot[id]
        return text_features, text_features_mask

    def __getitem__(self, idx):
        # Get the image
        #image = np.random.random((self.regions_in_image, self.visual_feature_dimension))
        image = np.load(self.image_features_dir + "{}.npy".format(self.image_ids[idx].split("#")[0])).reshape(
                       (self.regions_in_image, self.visual_feature_dimension))
        return to_tensor(image), to_tensor(self.all_text_features), self.all_text_features_mask, self.image_ids[idx]


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.img_one_hot = run(params.caption_file)
        self.train_ids = get_ids('train', params.split_file)
        self.val_ids = get_ids('val', params.split_file)
        self.plain_val_ids = get_ids('val', params.split_file, strip=True)
        self.test_ids = get_ids('test', params.split_file)
        self.plain_test_ids = get_ids('test', params.split_file, strip=True)
        self.regions_in_image = params.regions_in_image
        self.max_caption_len = params.max_caption_len
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
                                                                           params.image_features_dir,
                                                                           self.max_caption_len),
                                                            batch_size=1, shuffle=False, **kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(CustomDataSet1(self.img_one_hot,
                                                                           self.test_ids,
                                                                           self.plain_test_ids,
                                                                           params.regions_in_image,
                                                                           params.visual_feature_dimension,
                                                                           params.image_features_dir,
                                                                           self.max_caption_len),
                                                            batch_size=1, shuffle=False, **kwargs)

    def hard_negative_mining(self, model, pos_cap, pos_mask, pos_image, neg_cap, neg_mask, neg_image):
        model.eval()
        neg_mask = torch.autograd.Variable(neg_mask)
        pos_mask = torch.autograd.Variable(pos_mask)
        _, z_u, z_v = model(torch.autograd.Variable(neg_cap), neg_mask,
                            torch.autograd.Variable(pos_image), True)

        _, z_u_1, z_v_1 = model(torch.autograd.Variable(pos_cap), pos_mask,
                            torch.autograd.Variable(neg_image), True)

        bs = len(pos_image)

        hard_neg_cap = None
        hard_neg_mask = None
        hard_neg_img = None
        for i in range(bs):
            each_image = z_v[i]
            similarity = (each_image * z_u).sum(2).sum(1) / self.regions_in_image
            hardest_neg = similarity.max(dim=0)[1].data[0]
            if hard_neg_cap is None:
                hard_neg_cap = neg_cap[hardest_neg].unsqueeze(0)
                hard_neg_mask = neg_mask[hardest_neg].data.unsqueeze(0)
            else:
                hard_neg_cap = torch.cat((hard_neg_cap, neg_cap[hardest_neg].unsqueeze(0)), dim=0)
                hard_neg_mask = torch.cat((hard_neg_mask, neg_mask[hardest_neg].data.unsqueeze(0)), dim=0)

            each_cap = z_u_1[i]
            each_mask = pos_mask[i]
            similarity = (each_cap * z_v_1).sum(2).sum(1) / self.regions_in_image
            hardest_neg = similarity.max(dim=0)[1].data[0]
            if hard_neg_img is None:
                hard_neg_img = neg_image[hardest_neg].unsqueeze(0)
            else:
                hard_neg_img = torch.cat((hard_neg_img, neg_image[hardest_neg].unsqueeze(0)), dim=0)

        return pos_cap, pos_mask.data, pos_image, hard_neg_cap, hard_neg_mask, hard_neg_img
