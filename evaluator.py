from util import *
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
        self.idx_0, self.idx_1, self.idx_2, self.idx_3, self.idx_4 = self.create_indexes()

    def recall(self, model, is_test=False):
        """
        :param model:
        :param is_test:
        :return: Returns the recall at 1,5,10
        """
        model.eval()
        if is_test:
            ids = self.data_loader.test_ids
            plain_ids = self.data_loader.plain_test_ids
            data_loader = self.data_loader.test_data_loader
        else:
            ids = self.data_loader.val_ids
            plain_ids = self.data_loader.plain_val_ids
            data_loader = self.data_loader.eval_data_loader

        r_1 = 0
        r_5 = 0
        r_10 = 0
        for (caption, mask, images, label) in tqdm(data_loader):
            caption, mask, image = self.get_all_image_caption_pairs(caption, mask, images)
            s = model(to_variable(caption),
                      to_variable(mask),
                      to_variable(image),
                      False)
            similarity = s.data.cpu().numpy()

            # Compute similarity with the existing images
            top_10_img_idx = (-similarity[:]).argsort()[:10]
            if label[0] == plain_ids[top_10_img_idx[0]]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif label[0] in [plain_ids[x] for x in top_10_img_idx[1:5]]:
                r_5 += 1
                r_10 += 1
            elif label[0] in [plain_ids[x] for x in top_10_img_idx[6:10]]:
                r_10 += 1

        return r_1 / len(ids), r_5 / len(ids), r_10 / len(ids)

    def recall_i2t(self, model, is_test=False):
        """
        :param model:
        :param is_test:
        :return: Returns the recall at 1,5,10
        """
        model.eval()
        if is_test:
            ids = self.data_loader.test_ids
            plain_ids = self.data_loader.plain_test_ids
            data_loader = self.data_loader.test_data_loader
        else:
            ids = self.data_loader.val_ids
            plain_ids = self.data_loader.plain_val_ids
            data_loader = self.data_loader.eval_data_loader

        r_1 = 0
        r_5 = 0
        r_10 = 0
        for (images_, caption_, mask_, label) in tqdm(data_loader):
            similarity = None
            for i in range(5):
                caption, mask, image = self.get_all_image_caption_pairs_1(caption_, mask_, images_, i)
                s = model(to_variable(caption),
                          to_variable(mask),
                          to_variable(image),
                          False)
                if similarity is None:
                    similarity = s.data.cpu().numpy()
                else:
                    similarity = np.concatenate((similarity, s.data.cpu().numpy()), axis=0)

            # Compute similarity with the existing images
            top_10_img_idx = (-similarity[:]).argsort()[:10]
            if label[0] == ids[top_10_img_idx[0]].split("#")[0]:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif label[0] in [ids[x].split("#")[0] for x in top_10_img_idx[1:5]]:
                r_5 += 1
                r_10 += 1
            elif label[0] in [ids[x].split("#")[0] for x in top_10_img_idx[6:10]]:
                r_10 += 1

        return r_1 / len(plain_ids), r_5 / len(plain_ids), r_10 / len(plain_ids)

    @staticmethod
    def get_all_image_caption_pairs(caption, mask, images):
        images = to_tensor(images[0])
        caption = to_tensor(np.repeat(caption.numpy(), images.size(0), axis=0)).long()
        mask = to_tensor(np.repeat(mask.numpy(), images.size(0), axis=0))
        return caption, mask, images

    def get_all_image_caption_pairs_1(self, caption, mask, image, idx):
        captions = torch.index_select(caption[0], dim=0, index=self.get_idx(idx)).long()
        masks = torch.index_select(mask[0], dim=0, index=self.get_idx(idx)).float()
        image = to_tensor(np.repeat(image.numpy(), captions.size(0), axis=0))
        return captions, masks, image

    @staticmethod
    def create_indexes():
        idx_0 = np.arange(0, 1000)
        idx_1 = np.arange(1000, 2000)
        idx_2 = np.arange(2000, 3000)
        idx_3 = np.arange(3000, 4000)
        idx_4 = np.arange(4000, 5000)
        return to_tensor(idx_0).long(), to_tensor(idx_1).long(), to_tensor(idx_2).long(), to_tensor(idx_3).long(), \
               to_tensor(idx_4).long()

    def get_idx(self, idx):
        if idx == 0:
            return self.idx_0
        if idx == 1:
            return self.idx_1
        if idx == 2:
            return self.idx_2
        if idx == 3:
            return self.idx_3
        if idx == 4:
            return self.idx_4

    def avs(self, model):
        model.eval()
        queries = self.get_queries()

        # Write results in the file
        file = open("avs_results", "w", encoding="utf8")

        for qid, query in queries:
            similarity = None
            caption, mask = self.get_caption_and_mask(query)
            for image in tqdm(self.data_loader.iacc_data_loader):
                s = model(to_variable(caption),
                        to_variable(mask),
                        to_variable(image),
                        False)
                if similarity is None:
                    similarity = s.data.cpu().numpy()
                else:
                    similarity = np.concatenate((similarity, s.data.cpu().numpy()), axis=0)

            # Compute similarity with the existing images
            top_k_img_idx = (-similarity[:]).argsort()[:self.params.avs_k]
            for i, idx in enumerate(top_k_img_idx):
                file.write("1{}0 {} {} {} INF\n".format(qid, self.data_loader.shot_ids[idx], i + 1, 9999 - i))

        file.close()

    def get_queries(self):
        query_file = self.params.query_file
        # Loading the queries
        file = open(query_file, "r", encoding='utf8')
        queries = []
        for line in file.readlines():
            lines = line.split(":")
            query = lines[1].strip().lower()
            queries.append((lines[0], query))
        return queries

    def get_caption_and_mask(self, query):
        caption, _ = get_query_encoding(query)
        caption = np.tile(caption, (self.params.avs_bs, 1))
        mask = np.ones((self.params.avs_bs, caption.shape[1]))
        return to_tensor(caption).long(), to_tensor(mask).float()
