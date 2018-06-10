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
        for (images_, caption_, mask_, label) in tqdm(data_loader):
            found_in_1 = False
            found_in_5 = False
            found_in_10 = False
            for i in range(5):
                caption, mask, image = self.get_all_image_caption_pairs(caption_, mask_, images_, i)
                s = model(to_variable(caption),
                          to_variable(mask),
                          to_variable(image),
                          False)
                similarity = s.data.cpu().numpy()

                # Compute similarity with the existing images
                top_10_img_idx = ((-similarity[:]).argsort()[:10] * 5) + i
                if label[0] == ids[top_10_img_idx[0]].split("#")[0]:
                    found_in_1 = True
                    break
                elif label[0] in [ids[x].split("#")[0] for x in top_10_img_idx[1:5]]:
                    found_in_5 = True
                elif label[0] in [ids[x].split("#")[0] for x in top_10_img_idx[6:10]]:
                    found_in_10 = True

            if found_in_1:
                r_1 += 1
                r_5 += 1
                r_10 += 1
            elif found_in_5:
                r_5 += 1
                r_10 += 1
            elif found_in_10:
                r_10 += 1

        return r_1 / len(ids), r_5 / len(ids), r_10 / len(ids)

    def get_all_image_caption_pairs(self, caption, mask, image, idx):
        captions = torch.index_select(caption[0], dim=0, index=self.get_idx(idx)).long()
        masks = torch.index_select(mask[0], dim=0, index=self.get_idx(idx))
        image = to_tensor(np.repeat(image.numpy(), captions.size(0), axis=0))
        return captions, masks, image

    def create_indexes(self):
        idx_0 = np.zeros(len(self.data_loader.plain_val_ids))
        idx_0[range(0, len(self.data_loader.plain_val_ids), 5)] = 1
        idx_1 = np.zeros(len(self.data_loader.plain_val_ids))
        idx_1[range(1, len(self.data_loader.plain_val_ids), 5)] = 1
        idx_2 = np.zeros(len(self.data_loader.plain_val_ids))
        idx_2[range(2, len(self.data_loader.plain_val_ids), 5)] = 1
        idx_3 = np.zeros(len(self.data_loader.plain_val_ids))
        idx_3[range(3, len(self.data_loader.plain_val_ids), 5)] = 1
        idx_4 = np.zeros(len(self.data_loader.plain_val_ids))
        idx_4[range(4, len(self.data_loader.plain_val_ids), 5)] = 1
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