# SpectrumAugmenter
Performs data augmentation as according to the SpecAug paper. Modified from [Lingvo](https://github.com/tensorflow/lingvo).

Modified from [Lingvo](https://github.com/tensorflow/lingvo), test audio file is selected from [Sound Examples](http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/index-e.html).

## Requirements
- TensorFlow

For visualizing (option)
  - matplotlib
  - librosa
  - numpy

## How to use

```python
from __future__ import absolute_import, division, print_function

import librosa
import tensorflow as tf

from spectrum_augmenter import SpectrumAugmenter


if __name__ == '__main__':
    # Load an audio file as a floating point time series.
    audio, sampling_rate = librosa.load("test.wav")

    # Compute a mel-scaled spectrogram.
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)

    # (frequecy, time) -> (time, frequecy)
    mel_spectrogram = mel_spectrogram.transpose()

    # Inserts a dimension of 1 into a tensor's shape. 
    # (time, frequecy) -> (batch_size, time, frequecy)
    mel_spectrogram = mel_spectrogram.reshape(
        (1, mel_spectrogram.shape[0], mel_spectrogram.shape[1]))

    config = dict(
        # Maximum number of frequency bins of frequency masking.
        freq_mask_max_bins=30,
        # # Number of times we apply masking on the frequency axis.
        freq_mask_count=2,
        # Maximum number of frames of time masking. Overridden when use_dynamic_time_mask_max_frames = True.
        time_mask_max_frames=40,
        # Number of times we apply masking on the time axis. Acts as upper-bound when time_masks_per_frame > 0.
        time_mask_count=2,
        # Maximum number of frames for shifting in time warping.
        time_warp_max_frames=80,
    )

    specaug = SpectrumAugmenter(config)

    # (batch_size, time, frequecy)
    warped_masked_spectrogram = specaug(
        tf.convert_to_tensor(mel_spectrogram),
        tf.convert_to_tensor([mel_spectrogram.shape[0]]) # seq_len
    )


```

## Reference

[SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/pdf/1904.08779.pdf)
