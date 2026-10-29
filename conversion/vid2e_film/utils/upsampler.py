import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from . import Sequence
from .const import imgs_dirname
from .interpolator import Interpolator
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str, film_pair_batch_size: int = 1):
        assert os.path.isdir(input_dir), 'The input directory must exist'
        assert not os.path.exists(output_dir), 'The output directory must not exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir
        self.film_pair_batch_size = film_pair_batch_size

        path = os.path.join(os.path.dirname(__file__), "../film/film_net/Style/saved_model")
        self.interpolator = Interpolator(path, None)

    def upsample(self):
        sequence_counter = 0
        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath)
            if sequence is None:
                continue
            sequence_counter += 1
            pass  # sequence path logged by caller
            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_dirname)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)
        timestamps_list = list()

        idx = 0
        pair_buffer = []  # accumulate up to film_pair_batch_size pairs

        def flush_buffer(buf):
            nonlocal idx
            # Stack N pairs into batch tensors: [N, H, W, C]
            I0_batch = np.concatenate([p[0] for p in buf], axis=0)
            I1_batch = np.concatenate([p[1] for p in buf], axis=0)
            dt_batch = np.full(shape=(len(buf),), fill_value=0.5, dtype=np.float32)

            images_batch, F_0_1_batch, F_1_0_batch = self.interpolator.interpolate(
                I0_batch, I1_batch, dt_batch)

            for b_idx, (I0, I1, t0, t1) in enumerate([(p[0], p[1], p[2], p[3]) for p in buf]):
                image = images_batch[b_idx:b_idx+1]
                F_0_1 = F_0_1_batch[b_idx:b_idx+1] if F_0_1_batch.ndim == 4 else F_0_1_batch
                F_1_0 = F_1_0_batch[b_idx:b_idx+1] if F_1_0_batch.ndim == 4 else F_1_0_batch

                flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
                flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
                num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]) + 1e-6) / np.log(2)))

                if num_bisections <= 0:
                    total_frames = [image[0]]
                    total_timestamps_pair = [(t0 + t1) / 2]
                else:
                    left_imgs, left_ts = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
                    right_imgs, right_ts = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
                    total_timestamps_pair = left_ts + [(t0+t1)/2] + right_ts
                    total_frames = left_imgs + [image[0]] + right_imgs

                frames_with_t0 = [I0[0]] + total_frames
                timestamps_with_t0 = [t0] + total_timestamps_pair

                sorted_indices = np.argsort(timestamps_with_t0)
                frames_with_t0 = [frames_with_t0[j] for j in sorted_indices]
                timestamps_with_t0 = [timestamps_with_t0[j] for j in sorted_indices]

                timestamps_list.extend(timestamps_with_t0)
                for frame in frames_with_t0:
                    self._write_img(frame, idx, dest_imgs_dir)
                    idx += 1

            return I1_batch[-1], buf[-1][3]  # last I1 and t1 for final write

        last_I1 = None
        last_t1 = None

        for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__, disable=True):
            I0 = img_pair[0][None]
            I1 = img_pair[1][None]
            t0, t1 = time_pair
            pair_buffer.append((I0, I1, t0, t1))

            if len(pair_buffer) >= self.film_pair_batch_size:
                last_I1, last_t1 = flush_buffer(pair_buffer)
                pair_buffer = []

        if pair_buffer:
            last_I1, last_t1 = flush_buffer(pair_buffer)

        timestamps_list.append(last_t1)
        self._write_img(last_I1, idx, dest_imgs_dir)
        self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            return [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
            num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]))/np.log(2)))

            if num_bisections == 0:
                return [image[0]], [(t0 + t1) / 2]

        left_images, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        return images, timestamps

    def _prepare_output_dir(self, src_dir: str, dest_dir: str):
        # Copy directory structure.
        def ignore_files(directory, files):
            return [f for f in files if os.path.isfile(os.path.join(directory, f))]
        shutil.copytree(src_dir, dest_dir, ignore=ignore_files)

    @staticmethod
    def _write_img(img: np.ndarray, idx: int, imgs_dir: str):
        assert os.path.isdir(imgs_dir)
        img = np.clip(img * 255, 0, 255).astype("uint8")
        path = os.path.join(imgs_dir, "%08d.png" % idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)

    @staticmethod
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        with open(timestamps_filename, 'w') as t_file:
            t_file.writelines([str(t) + '\n' for t in timestamps])