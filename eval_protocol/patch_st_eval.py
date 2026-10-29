import sys
p = sys.argv[1]
s = open(p).read()
add = """DATASETS_INFO = {
    'MEVIS_U_PSEUDO': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/mevis_valid_u_10ms',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json',
        'mask_file': None,
    },
    'MOUSESIS_PSEUDO': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/mousesis_10ms',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_converted/meta_expressions.json',
        'mask_file': None,
    },
    'MEVIS_U_E2VID': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/mevis_valid_u',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json',
        'mask_file': None,
    },
    'MOUSESIS_E2VID': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/mousesis',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_converted/meta_expressions.json',
        'mask_file': None,
    },
    'MOUSESIS_PSEUDO75': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/mousesis_75ms',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_converted/meta_expressions.json',
        'mask_file': None,
    },
    'NCALT_PSEUDO': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/ncalt_hf_10ms',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech_converted_hf/meta_expressions.json',
        'mask_file': None,
    },
    'MOUSESIS_TRAIN_PSEUDO75': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/mousesis_train_75ms',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_train_converted/meta_expressions.json',
        'mask_file': None,
    },
    'MOUSESIS_TRAIN_E2VID': {
        'image_folder': '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/mousesis_train',
        'expression_file': '/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_train_converted/meta_expressions.json',
        'mask_file': None,
    },
"""
assert s.count("DATASETS_INFO = {\n") == 1
s = s.replace("DATASETS_INFO = {\n", add, 1)
old = "    for item in tqdm.tqdm(dataloader):\n"
assert s.count(old) == 1
s = s.replace(old, old + "        if os.path.isdir(os.path.join(work_dir, 'Annotations', item['video_id'], item['exp_id'])):\n            continue  # resume\n", 1)
assert s.count("use_flash_attn=True") == 1
s = s.replace("use_flash_attn=True", "use_flash_attn=False", 1)  # container ships flash_attn 2.0.4
PREFIX = ('import sys\n'
          'sys.path.insert(0, "/users/sfedorchenko/Event-SAMA-train/projects/llava_sam2/evaluation")  # RES.py: from utils import REFER\n')
s = PREFIX + s
open(p, "w").write(s)
