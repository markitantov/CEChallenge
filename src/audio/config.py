import sys

sys.path.append('../src')

from models.audio_expr_models import *

c_config: dict = {
    'ABAW_WAV_ROOT': '/media/maxim/Databases/ABAW2024/data/wavs',
    'ABAW_FILTERED_WAV_ROOT': '/media/maxim/Databases/ABAW2024/data/vocals',
    'ABAW_VIDEO_ROOT': '/media/maxim/Databases/ABAW2024/data/videos',
    'ABAW_LABELS_ROOT': '/media/maxim/Databases/ABAW2024/6th_ABAW_Annotations/EXPR_Classification_Challenge',
    'ABAW_FEATURES_ROOT': '/media/maxim/Databases/ABAW2024/features/open_mouth',

    'MELD_WAV_ROOT': '/media/maxim/Databases/MELD.Raw/wavs',
    'MELD_FILTERED_WAV_ROOT': '/media/maxim/Databases/MELD.Raw/vocals',
    'MELD_LABELS_ROOT': '/media/maxim/Databases/MELD.Raw/labels',
    'MELD_VAD_ROOT': '/media/maxim/Databases/MELD.Raw/',
    
    'C_WAV_ROOT': '/media/maxim/Databases/C-EXPR-DB/wavs/',
    'C_FILTERED_WAV_ROOT': '/media/maxim/Databases/C-EXPR-DB/vocals/',
    'C_VIDEO_ROOT': '/media/maxim/Databases/C-EXPR-DB/videos/',
    'C_LABELS_ROOT': None,
    'C_FEATURES_ROOT': '/media/maxim/Databases/C-EXPR-DB/features/open_mouth/',
    
    ###
    'LOGS_ROOT': '/media/maxim/WesternDigital/ABAWLogs/C',
    'MODEL_PARAMS': {
        'model_cls': ExprModelV3,
        'args': {
            'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
        }
    },
    'FILTERED': False,
    'AUGMENTATION': False,
    'NUM_EPOCHS': 100,
    'BATCH_SIZE': 24,
}