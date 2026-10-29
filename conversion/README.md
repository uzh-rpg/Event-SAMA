# Event-SAMA: RGB to Event Conversion

This repository contains the pipeline to reproduce the synthetic event data used in the paper *Event-SAMA: Language-Grounded Video Object Segmentation on Neuromorphic Event Streams*.

## Setup

```bash
git clone <repo-url> Event-SAMA
cd Event-SAMA
docker build -t event-sama .
```

FILM weights (only needed for ChatUniVi):

```bash
wget https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip -O /tmp/pretrained_models.zip
unzip /tmp/pretrained_models.zip 'pretrained_models/film_net/Style/*' -d /tmp
mv /tmp/pretrained_models/film_net vid2e_film/film/
```

Start the container (`--gpus all` is only needed for FILM):

```bash
docker run --gpus all -it --rm -v "$PWD":/workspace -v /path/to/your/data:/data event-sama
```

## Run

```bash
python convert.py --config configs/mevis.yaml --input /data/mevis --output /data/events \
    --workers 8 --limit 10 --joblist joblist.txt --tmpdir /tmp/film --eval eval/mevis
```

Only `--config`, `--input` and `--output` are required. The configs used to convert the data can all be found under /config

## License

This repository is released under the Apache License 2.0, **except** `vid2e_film/`, which is adapted from
[vid2e](https://github.com/uzh-rpg/rpg_vid2e) and remains under the
[GNU GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html). 

Generated event data inherits the terms of the respective source dataset.

## Citation

If you use this code or data, please cite our paper and the methods it builds on:

```bibtex
@inproceedings{ziegler2023realtime,
  title     = {Real-time event simulation with frame-based cameras},
  author    = {Ziegler, Andreas and Teigland, Daniel and Tebbe, Jonas and Gossard, Thomas and Zell, Andreas},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023}
}
@inproceedings{gehrig2020video,
  title     = {Video to Events: Recycling Video Datasets for Event Cameras},
  author    = {Gehrig, Daniel and Gehrig, Mathias and Hidalgo-Carri{\'o}, Javier and Scaramuzza, Davide},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
@inproceedings{reda2022film,
  title     = {FILM: Frame Interpolation for Large Motion},
  author    = {Reda, Fitsum and Kontkanen, Janne and Tabellion, Eric and Sun, Deqing and Pantofaru, Caroline and Curless, Brian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
@inproceedings{kroeger2016fast,
  title     = {Fast Optical Flow using Dense Inverse Search},
  author    = {Kroeger, Till and Timofte, Radu and Dai, Dengxin and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2016}
}
```
