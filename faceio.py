import os, wave

# 48000kHz(major), 25fps(?)
def get_date_directories(kind,RATIO=40,hidden_unfinished=False):
    ROOT_PATH = '/home/yunyangli/CMLR/%s/' % kind
    AUDIO_PATH = ROOT_PATH.replace('video', 'audio')
    TARGET_PATH = '/home/yunyangli/CMLR/%s_feat/' % kind
    VIDEO_FRAMERATE = 25

    results = []

    for speaker in sorted(os.listdir(ROOT_PATH)):
        speaker_path = os.path.join(ROOT_PATH, speaker)
        os.makedirs(os.path.join(TARGET_PATH, speaker), exist_ok=True)
        for idx, date in enumerate(sorted(os.listdir(speaker_path))):
            if idx % RATIO == 0:
                date_path = os.path.join(speaker_path, date)
                o = {'sources': []}
                target_path = os.path.join(TARGET_PATH, speaker, '%s.npy' % date)
                if hidden_unfinished and not os.path.exists(target_path):
                    continue
                for filename in sorted(os.listdir(date_path)):
                    if not filename.endswith('wav') and not filename.endswith('mp4'):
                        continue
                    p = os.path.join(AUDIO_PATH, speaker, date, filename.replace('mp4', 'wav'))
                    wav = wave.open(p, 'rb')
                    num_frame = wav.getnframes()
                    slice_num = int(num_frame / wav.getframerate() * VIDEO_FRAMERATE)
                    o['sources'].append((os.path.join(date_path, filename), slice_num))
                o['target'] = target_path
                results.append(o)
    return results