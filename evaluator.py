from util import *
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

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
                      True)
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

    @staticmethod
    def get_all_image_caption_pairs(caption, mask, images):
        images = to_tensor(images[0])
        caption = to_tensor(np.repeat(caption.numpy(), images.size(0), axis=0)).long()
        mask = to_tensor(np.repeat(mask.numpy(), images.size(0), axis=0))
        return caption, mask, images
