"""Event-native counterpart of ref_vos_eval_st.py: runs VOS inference with
event-voxel input (via EventRefVOSDataset + model.predict_forward_event)
instead of RGB JPEGImages, so ablation checkpoints can be evaluated on the
same modality they were trained on."""
import argparse
import json
import os

print("[PATCH] Disabling Flash Attention 2...", flush=True)
from transformers.modeling_utils import PreTrainedModel


@classmethod
def _force_eager_attn_implementation(cls, config, *args, **kwargs):
    if hasattr(config, '_attn_implementation_internal'):
        config._attn_implementation_internal = 'eager'
    config._attn_implementation = 'eager'
    return config


PreTrainedModel._autoset_attn_implementation = _force_eager_attn_implementation
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
    lambda cls, config, *a, **kw: config
)
print("[PATCH] Attention implementation forced to eager (no FA2)", flush=True)

import mmengine
import numpy as np

import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer

from projects.llava_sam2.evaluation.dataset import EventRefVOSDataset
from projects.llava_sam2.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    from PIL import Image
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


# Event-converted eval splits. MEVIS_U_EVENT: full 50/50 coverage, exact
# same conversion pipeline/directory the live training config uses
# (CAPSTOR_TRANSFER + '/mevis'). REVOS_EVENT: partial coverage (329/416 --
# MOSE/OVIS/UVO fully converted, LV-VIS/test not converted); the dataset's
# own pre-filter drops the uncovered videos so this evaluates on whatever
# real event data exists, not a guess.
CAPSTOR_V2 = '/capstor/scratch/cscs/mzehnde/sama_converted_v2'
CAPSTOR_REVOS = '/capstor/scratch/cscs/mzehnde/sama_converted_revos'
CAPSTOR_TRANSFER = '/capstor/scratch/cscs/mzehnde/sama_converted_transfer/max_sergei'

DATASETS_INFO = {
    'MEVIS_U_EVENT': {
        'event_folder': CAPSTOR_TRANSFER + '/mevis',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json',
        'mask_file': '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/mask_dict.json',
        'event_layout': 'nested',
        'resolver': None,
    },
    'REVOS_EVENT': {
        'event_folder': CAPSTOR_V2 + '/revos_uvo/all',  # UVO fallback root
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/video_datas/revos/meta_expressions_valid_.json',
        'mask_file': '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/video_datas/revos/mask_dict.json',
        'event_layout': 'nested',
        'resolver': 'revos_uvo_prefix',
    },
}


def _make_revos_resolver(uvo_root):
    """Same prefix-based resolution used by _UVOPathMixin at training time:
    LV-VIS/OVIS/TAO/MOSE keys go to their standalone event roots; anything
    else falls back to the UVO <yt>.f<segment> remap."""
    import re
    lvvis_root = CAPSTOR_V2 + '/lvvis'
    ovis_root = CAPSTOR_V2 + '/revos_ovis'
    tao_root = CAPSTOR_V2 + '/revos_tao'
    mose_root = CAPSTOR_REVOS + '/revos_mose'
    prefix_roots = [('LV-VIS/', lvvis_root), ('OVIS/', ovis_root),
                    ('TAO/', tao_root), ('MOSE/', mose_root)]

    suffix_re = re.compile(r'\.f\d+$')
    uvo_map = {}
    if os.path.isdir(uvo_root):
        for d in os.listdir(uvo_root):
            if os.path.isfile(os.path.join(uvo_root, d, 'events.h5')):
                base = suffix_re.sub('', d)
                uvo_map.setdefault(base, d)
                uvo_map[d] = d

    def resolve(video_id):
        vid = str(video_id)
        for prefix, root in prefix_roots:
            if vid.startswith(prefix):
                parts = vid.split('/')
                return os.path.join(root, *parts[1:], 'events.h5')
        key = vid.split('/')[-1]
        dirname = uvo_map.get(key) or uvo_map.get(vid)
        if dirname is None:
            return os.path.join(uvo_root, vid, 'events.h5')  # deliberately missing -> pre-filter drops it
        return os.path.join(uvo_root, dirname, 'events.h5')

    return resolve


def parse_args():
    parser = argparse.ArgumentParser(description='Event RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS_U_EVENT',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true', default=True)
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None)  # dummy
    parser.add_argument('--limit', type=int, default=None,
                        help='Smoke-test: only evaluate the first N (video, '
                             'expression) items.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = 'work_dirs/foobar'

    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]

    resolver = None
    if dataset_info['resolver'] == 'revos_uvo_prefix':
        resolver = _make_revos_resolver(dataset_info['event_folder'])

    dataset = EventRefVOSDataset(
        event_folder=dataset_info['event_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
        event_layout=dataset_info['event_layout'],
        event_path_resolver=resolver,
    )
    if args.limit is not None:
        dataset.text_data = dataset.text_data[:args.limit]
        print(f"[SMOKE TEST] limiting to first {len(dataset.text_data)} items", flush=True)

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            if '<image>' not in item['text_prompt']:
                item['text_prompt'] = "<image>" + '\n' + item['text_prompt']
            result = model.predict_forward_event(
                event_frames=item['event_frames'],
                text=item['text_prompt'],
                tokenizer=tokenizer,
                prediction_only=False,
                max_frames=32,
            )

        text_idx = 0
        text_prediction = result['prediction']
        if 'prediction_masks' in result and result['prediction_masks'] is not None and len(result['prediction_masks']) > 0:
            mask_prediction = result['prediction_masks'][text_idx]
        else:
            print("[INFO] Question: {}".format(item['text_prompt']))
            print("[INFO] Text_prediction: {}".format(text_prediction))
            mask_prediction = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        if args.submit:
            async_func(executor, mask_save, item=item, mask_prediction=mask_prediction, work_dir=work_dir)
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        result = {
            'index': item['index'],
            'video_id': item['video_id'],
            'exp_id': item['exp_id'],
            'text_prediction': text_prediction,
            'frames': item['frames'],
            'exp': item['text_prompt'],
            'prediction_masks': encoded_mask,
        }
        results.append(result)

    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')

    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f'{work_dir}/results.json', 'w'))

    if rank == 0:
        print('Done')
