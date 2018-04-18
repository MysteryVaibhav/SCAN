import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SCAN(torch.nn.Module):
    def __init__(self, params):
        super(SCAN, self).__init__()
        self.regions_in_image = params.regions_in_image
        self.word_embeddings = nn.Embedding(params.vocab_size, params.embedding_dimension)
        nn.init.xavier_uniform(self.word_embeddings.weight)
        self.linear_transform = nn.Linear(in_features=params.visual_feature_dimension, out_features=params.hidden_dimension)
        self.text_encoder = Encoder(params.hidden_dimension, params.embedding_dimension)
        self.sca = StackedCrossAttention(params.lambda_1)

    def forward(self, input_caption, mask, input_image, is_inference):
        # Sorting batch by length of captions, can't do this before because of negative mining
        #seq_lens = mask.sum(dim=1).long()
        #seq_lens, perm_idx = seq_lens.sort(0, descending=True)

        # sort everything according to permutation idx
        #input_caption = input_caption[perm_idx]
        #mask = mask[perm_idx]
        #input_image = input_image[perm_idx]

        embeds = self.word_embeddings(input_caption)                                            # bs * max_seq_len * 300
        h_t = self.text_encoder(embeds, mask)
        # Modify mask, according to max_seq_len in the batch
        #max_seq_len_in_batch = h_t.size(1)
        #mask = mask[:, :max_seq_len_in_batch]

        # Projecting the image to embedding dimensions
        h = self.linear_transform(input_image)                                        # bs * roi * hidden_dim

        similarity_matrix = torch.bmm(F.normalize(input=h, p=2, dim=2),
                                      F.normalize(input=h_t,
                                                  p=2, dim=2).permute(0, 2, 1))                                  # bs * roi * max_seq_len
        s_t = F.normalize(input=similarity_matrix.clamp(min=0), p=2, dim=2)
        a_v = self.sca(h_t, mask, h, s_t)                                             # bs * max_seq_len * hidden_dim

        # Average pooling for image_2_text
        #s_t_i = F.cosine_similarity(h_t, a_v, dim=2).sum(1) / len(h_t)
        s_t_i = F.cosine_similarity(h_t * mask.unsqueeze(2), a_v * mask.unsqueeze(2), dim=2).sum(1) / mask.sum(dim=1)
        
        # re-order the similarity scores
        return s_t_i


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dimension
        self.gru = nn.GRU(embedding_dimension, hidden_dimension)

    def forward(self, embeds, mask):
        # Sorting sequences by their lengths for packing
        embeds = embeds.permute(1, 0, 2)                                                         # seq_len * batch_size * embedding_dimension

        #packed_input = pack_padded_sequence(embeds, seq_lens.data.cpu().numpy())
        h, _ = self.gru(embeds)
        #h, _ = pad_packed_sequence(packed_outputs)

        # h = ( h_forward + h_backward ) / 2
        #h = h.view(h.size(0), h.size(1), 2, -1).sum(2).view(h.size(0), h.size(1), -1) / 2

        return h.permute(1, 0, 2)                                                               # bs * max_seq_len * hidden_dim


class StackedCrossAttention(torch.nn.Module):
    def __init__(self, lambda_1):
        super(StackedCrossAttention, self).__init__()
        self.lambda_1 = lambda_1

    def forward(self, h, mask, img, s_t):
        alpha_t = self.lambda_1 * s_t
        alpha_t.data.masked_fill_((1 - mask).data.unsqueeze(1).byte(), -float('inf'))
        alpha = F.softmax(input=alpha_t, dim=1)                                                 # bs * roi * max_seq_len
        alpha.data.masked_fill_((1 - mask).data.unsqueeze(1).byte(), 0)
        attended_image_vectors = torch.bmm(alpha.permute(0, 2, 1), img)                         # bs * max_seq_len * hidden_dim
        return attended_image_vectors


