import os


REMOVE_STOP_WORDS = False
# TRAINING PARAMS
BATCH_SIZE = 128
EPOCHS = 30
CLIP_VALUE = 2
LEARNING_RATE = 0.0002

# For Bi-LSTM
EMBEDDING_DIMENSION = 300
HIDDEN_DIMENSION = 512
VOCAB_SIZE = 7737 #6452
MAX_CAPTION_LEN = 82 #49

# DATA RELATED
VISUAL_FEATURE_DIMENSION = 2048
NO_OF_REGIONS_IN_IMAGE = 72

# MODEL 
MARGIN = 0.2

# PATH
#TRAIN_IMAGES_DIR = '/mnt/ssd1/junweil/vision_language/resnet-152/'
TRAIN_IMAGES_DIR = '/mnt/ssd1/junweil/vision_language/poyao_bottomup_feats_72/'
CAPTION_INFO = 'results_20130124.token'
SPLIT_INFO = '/mnt/ssd1/junweil/vision_language/splits/'
IMAGES_DIR = '/data/disk1/junweil/vision_language/data/flickr30k/flickr30k_images/'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#Local path for testing
#CAPTION_INFO = 'C:\\Users\\myste\\Downloads\\results_20130124.token'
#SPLIT_INFO = 'C:\\Users\\myste\\Downloads\\split\\'
#CONCEPT_DIR = 'C:\\Users\\myste\\Downloads\\semantic_feat\\'