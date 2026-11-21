#!/usr/bin/env python3
"""Prepare patient-balanced training references; run two resumable GPU shards."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import his_mmdm_he_augment as core

LABELS = ('methylated', 'unmethylated')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_for(seed, *parts):
    value = '|'.join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') % (2**63)


def quotas(capacities, total, seed):
    """Water-fill quotas over all patients, without selecting with replacement."""
    patients = sorted(capacities)
    random.Random(seed).shuffle(patients)
    if total < len(patients):
        raise ValueError('The source budget must cover every eligible patient in its class.')
    if sum(capacities.values()) < total:
        raise ValueError(f'Not enough distinct tissue patches: need {total}, have {sum(capacities.values())}.')
    result = dict.fromkeys(patients, 0)
    while sum(result.values()) < total:
        for patient in patients:
            if result[patient] < capacities[patient]:
                result[patient] += 1
                if sum(result.values()) == total:
                    break
    return result


def choose_sources(tiles, labels, per_class, seed, fold, loader):
    chosen = []
    summaries = []
    for label in LABELS:
        group = tiles[tiles.patient_id.map(labels).eq(label)]
        capacities = group.groupby('patient_id').size().to_dict()
        allocation = quotas(capacities, per_class, seed_for(seed, label, 'patient allocation'))
        by_patient = {}
        for patient, count in allocation.items():
            sample = loader.training_sample(group, patient, epoch=0, seed=seed_for(seed, label), count=count)
            records = sample.to_dict('records')
            if len({r['patch_id'] for r in records}) != count:
                raise ValueError('Sampling unexpectedly duplicated a source patch.')
            by_patient[patient] = records
            summaries.append(dict(mgmt_patient_label=label, patient_id=patient, selected_sources=count))
        # Interleave patients. Balance each shard by class while spreading each patient's patches.
        shard_counts = [0, 0]
        patient_shards = defaultdict(lambda: [0, 0])
        while any(by_patient.values()):
            for patient in allocation:
                if not by_patient[patient]:
                    continue
                row = by_patient[patient].pop(0)
                available = [i for i in range(2) if shard_counts[i] < per_class // 2]
                shard = min(available, key=lambda i: (patient_shards[patient][i], shard_counts[i], i))
                row['mgmt_patient_label'] = label
                row['workload_shard'] = shard
                patient_shards[patient][shard] += 1
                shard_counts[shard] += 1
                chosen.append(row)
    if len({r['patch_id'] for r in chosen}) != len(chosen):
        raise ValueError('Duplicate patch IDs across classes.')
    return chosen, summaries


def prepare(args):
    from PIL import Image
    sys.path.insert(0, str(args.repo.resolve() / 'data_prep'))
    import common
    import loader
    core.require_new(args.out)
    if loader.folder_kind(args.package) != 'package':
        raise ValueError('Use the verified portable package.')
    config_path = args.repo / 'configs/tiling.yaml'
    cfg, cfg_hash, _ = common.load_tiling_config(config_path)
    folds_path = args.repo / 'manifests/patient_folds.csv'
    manifest_path = args.package / loader.PACKAGE_MANIFEST
    manifest = core.read_json(manifest_path)
    if (cfg['tile_px'], cfg['target_mpp']) != (224, 0.5):
        raise ValueError('Expected native 224-pixel package at 0.5 um/px.')
    if manifest['folds_sha256'] != core.digest(folds_path) or manifest['config_sha256'] != cfg_hash:
        raise ValueError('Package fold map or tiling configuration differs from the checkout.')
    label_rows = core.unique_by(core.read_csv(args.labels, ('patient_id',)), 'patient_id')
    labels = {p: core.label_status(row) for p, row in label_rows.items()}
    tiles = loader.load_modeling(args.package)
    if tiles.patch_id.duplicated().any():
        raise ValueError('The modeling list contains duplicate patch IDs.')
    # Preserve tissue/edge semantics before with_roles replaces the role column.
    tiles = tiles[tiles.source.eq('TCGA') & tiles.role.eq('tissue')].copy()
    folds = common.read_folds(folds_path)
    if not tiles.patient_id.map(lambda p: folds.get(('TCGA', p), 'missing')).eq(tiles.fold).all():
        raise ValueError('Modeling folds disagree with the recorded fold map.')
    tiles = loader.with_roles(tiles, args.fold)
    tiles = tiles[tiles.role.eq('train')]
    missing_labels = set(tiles.patient_id) - set(labels)
    if missing_labels:
        raise ValueError(f'Missing molecular labels for training patients: {sorted(missing_labels)}')
    chosen, allocation = choose_sources(tiles, labels, args.sources_per_class, args.seed, args.fold, loader)
    for label in LABELS:
        group = [r for r in chosen if r['mgmt_patient_label'] == label]
        print(f'{label}: {len(group)} sources from {len({r["patient_id"] for r in group})} patients', flush=True)
    # Select the repeat-check subset across as many patients as its budget permits.
    repeat_ids = set()
    for label in LABELS:
        group = [r for r in chosen if r['mgmt_patient_label'] == label]
        patients = sorted({r['patient_id'] for r in group})
        random.Random(seed_for(args.seed, label, 'repeat')).shuffle(patients)
        queues = {p: [r for r in group if r['patient_id'] == p] for p in patients}
        while len([r for r in group if r['patch_id'] in repeat_ids]) < min(args.verify_per_class, len(group)):
            for p in patients:
                if queues[p]:
                    repeat_ids.add(queues[p].pop(0)['patch_id'])
                    if len([r for r in group if r['patch_id'] in repeat_ids]) == min(args.verify_per_class, len(group)):
                        break
    index_path = args.root / 'patches' / cfg['version'] / f'index_{args.subset}.csv'
    index = common.read_rows(index_path)
    raw_csvs = {(r['source'], r['slide_id']): args.root / r['csv'] for r in index}
    wanted = defaultdict(dict)
    for row in chosen:
        wanted[(row['source'], row['slide_id'])][row['patch_id']] = row
    scores = {}
    score_hashes = {}
    for key, requested in wanted.items():
        path = raw_csvs[key]
        score_hashes[str(path.resolve())] = core.digest(path)
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['patch_id'] in requested:
                    if row['pixel_sha256'] != requested[row['patch_id']]['pixel_sha256']:
                        raise ValueError(f'Score pixel hash mismatch: {row["patch_id"]}')
                    scores[row['patch_id']] = row
        if not requested.keys() <= scores.keys():
            raise ValueError(f'Missing source scores for {key}')
    tracked = {'folds': folds_path, 'tiling': config_path, 'settings': manifest_path,
               'modeling': args.package / 'modeling_patches.csv', 'labels': args.labels}
    hashes = {name: core.digest(path) for name, path in tracked.items()}
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        staging = Path(temporary)
        references = staging / 'references'
        (references / 'originals').mkdir(parents=True)
        shutil.copyfile(folds_path, references / 'patient_folds.csv')
        shutil.copyfile(args.labels, references / 'patient_labels.csv')
        refs = []
        with loader.TileReader(args.package, tiling_config=config_path) as reader:
            for i, row in enumerate(chosen, 1):
                ref = f'ref{i:06d}'
                original = reader.read(row)
                if original.shape != (224, 224, 3) or original.dtype.name != 'uint8':
                    raise ValueError('Expected an RGB uint8 224-pixel source.')
                rel = f'originals/{ref}.png'
                Image.fromarray(original).save(references / rel)
                scored = scores[row['patch_id']]
                qc_fields = dict(morphology='', qc_defect_flags='', tissue_fraction=scored['mask_fraction'],
                                 laplacian_variance=scored['laplacian_variance'],
                                 near_white_fraction=scored['near_white_fraction'],
                                 near_black_fraction=scored['near_black_fraction'])
                core.check_qc({'patch_id': row['patch_id'], **qc_fields}, review_pending=True)
                refs.append(dict(reference_id=ref, source='TCGA', patient_id=row['patient_id'],
                    assigned_fold=row['fold'], slide_id=row['slide_id'], patch_id=row['patch_id'],
                    image_path=rel, source_sha256=core.digest(references / rel),
                    pixel_sha256=row['pixel_sha256'], mgmt_patient_label=row['mgmt_patient_label'],
                    source_mpp=0.5, source_review_status='pending', workload_shard=row['workload_shard'],
                    source_x_target=row['x_target'], source_y_target=row['y_target'],
                    repeat_checks=row['patch_id'] in repeat_ids, **qc_fields))
        core.save_csv(references / 'input_manifest.csv', refs)
        core.save_csv(staging / 'patient_allocation.csv', allocation)
        core.save_json(references / 'references.json', dict(
            held_out_fold=args.fold, batch_mode=True, preview_only=True,
            manifest_sha256=core.digest(references / 'input_manifest.csv'), inputs_sha256=hashes,
            modeling_folder=str(args.package.resolve()), prepared_at=core.stamp()))
        core.save_json(staging / 'batch_plan.json', dict(
            protocol='frozen-hmu1st-he-resize-v1', held_out_fold=args.fold, seed=args.seed, eta=0.3,
            source_count=len(refs), variants_per_source=2, synthetic_count=len(refs)*2,
            sources_per_class=args.sources_per_class, shard_count=2, verify_per_class=args.verify_per_class,
            source_budget='TCGA training tissue only; patient-balanced within MGMT class',
            references_sha256=core.digest(references / 'references.json'),
            allocation_sha256=core.digest(staging / 'patient_allocation.csv'),
            source_score_hashes=score_hashes, created_at=core.stamp(),
            source_review_status='pending', review_status='pending', training_eligible=False))
        for name, path in tracked.items():
            core.require_hash(path, hashes[name])
        for path, sha in score_hashes.items():
            core.require_hash(Path(path), sha)
        staging.rename(args.out)
    atomic_json(args.out.parent / 'latest_batch.json', dict(batch=str(args.out.resolve())))
    print(f'Prepared {len(refs)} sources for {len(refs)*2} variants: {args.out}', flush=True)


def validate_batch(batch):
    plan = core.read_json(batch / 'batch_plan.json')
    refs = batch / 'references'
    prep = core.read_json(refs / 'references.json')
    core.require_hash(refs / 'references.json', plan['references_sha256'])
    core.require_hash(refs / 'input_manifest.csv', prep['manifest_sha256'])
    core.require_hash(batch / 'patient_allocation.csv', plan['allocation_sha256'])
    core.require_hash(refs / 'patient_labels.csv', prep['inputs_sha256']['labels'])
    rows = core.read_csv(refs / 'input_manifest.csv', ('reference_id', 'workload_shard', 'repeat_checks'))
    if (len(rows) != plan['source_count'] or plan['variants_per_source'] != 2 or plan['shard_count'] != 2
            or plan['eta'] != 0.3 or plan['synthetic_count'] != 2*len(rows)):
        raise ValueError('Inconsistent batch dimensions or inference configuration.')
    expected = plan['sources_per_class']
    if Counter(r['mgmt_patient_label'] for r in rows) != Counter(dict.fromkeys(LABELS, expected)):
        raise ValueError('The two MGMT source classes are not balanced.')
    labels = {r['patient_id']: core.label_status(r) for r in core.read_csv(refs / 'patient_labels.csv', ('patient_id',))}
    for row in rows:
        if row['mgmt_patient_label'] != labels[row['patient_id']] or row['workload_shard'] not in {'0', '1'}:
            raise ValueError('Invalid patient label or shard assignment.')
    for shard in ('0', '1'):
        if Counter(r['mgmt_patient_label'] for r in rows if r['workload_shard'] == shard) != Counter(dict.fromkeys(LABELS, expected//2)):
            raise ValueError('Each shard must have equal source counts from both classes.')
    expected_repeats=min(plan['verify_per_class'],expected)
    if Counter(r['mgmt_patient_label'] for r in rows if r['repeat_checks']=='True') != Counter(dict.fromkeys(LABELS,expected_repeats)):
        raise ValueError('Repeat-check subset differs from the saved plan.')
    return plan, rows


def verified_receipt(folder, ref, signature):
    path = folder / 'records' / f'{ref}.json'
    if not path.exists():
        return None
    receipt = core.read_json(path)
    if receipt['signature_sha256'] != signature or receipt['reference_id'] != ref or len(receipt['outputs']) != 2:
        raise ValueError(f'Resume record differs from this run: {ref}')
    if {r['variant_id'] for r in receipt['outputs']}!={f'{ref}-01',f'{ref}-02'} or any(r['reference_id']!=ref for r in receipt['outputs']):
        raise ValueError(f'Variant identities differ from the source receipt: {ref}')
    for rel, sha in receipt['file_sha256'].items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe path in completion receipt.')
        core.require_hash(folder / path, sha)
    return receipt


def source_order(rows):
    ordered=[]
    for repeat in ('True','False'):
        groups={label:sorted((r for r in rows if r['repeat_checks']==repeat and r['mgmt_patient_label']==label),
                             key=lambda r:r['reference_id']) for label in LABELS}
        for i in range(max(map(len,groups.values()),default=0)):
            for label in LABELS:
                if i<len(groups[label]):ordered.append(groups[label][i])
    if len(ordered)!=len(rows):raise ValueError('Invalid repeat-check assignment.')
    return ordered


def worker(args):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    batch = args.batch.resolve()
    plan, all_rows = validate_batch(batch)
    args.refs = batch / 'references'
    args.fold = plan['held_out_fold']
    args.phase = 'batch'
    config = core.release_settings(args.upstream.resolve())
    checkpoint, weights, all_rows, prep, allowed = core.validated_inputs(args)
    selected = [r for r in all_rows if r['workload_shard'] == str(args.shard)]
    selected = source_order(selected)
    folder = batch / f'generated/shard-{args.shard}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError(f'A worker already holds shard {args.shard}.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    sys.path.insert(0, str(args.upstream.resolve()))
    from cg_diffusion.image_datasets import center_crop_arr
    from cg_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this workload.')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    hardware = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'],
                               text=True, capture_output=True, check=True).stdout.strip()
    runtime = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                   numpy=np.__version__, pillow=Image.__version__, cudnn=torch.backends.cudnn.version(),
                   gpu=torch.cuda.get_device_name(device), compute_capability=list(torch.cuda.get_device_capability(device)),
                   cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    signature = dict(batch_plan_sha256=core.digest(batch/'batch_plan.json'),
        input_manifest_sha256=prep['manifest_sha256'], checkpoint_sha256=weights['sha256'],
        wrapper_sha256=core.digest(Path(core.__file__)), batch_script_sha256=core.digest(Path(__file__)),
        upstream_commit=core.COMMIT, model_flags=config, runtime=runtime, shard=args.shard,
        encode_eta=0, decode_eta=0.3, steps=1000, source_class_id=10, target_class_id=10,
        source_mpp=0.5, original_px=224, native_px=128, classifier_px=224,
        classifier_resize='128 to 224 bicubic', frozen=True, classifier_guidance=False)
    signature_sha = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    config_path = folder/'batch_run.json'
    if config_path.exists():
        if core.read_json(config_path)['signature_sha256'] != signature_sha:
            raise ValueError('Existing shard used different inputs, code, model flags or software/GPU type.')
    else:
        atomic_json(config_path, dict(phase='batch', signature=signature, signature_sha256=signature_sha,
                                     created_at=core.stamp(), hardware=hardware,
                                     review_status='pending', training_eligible=False))
    print(f'Shard {args.shard} | {len(selected)} sources | {2*len(selected)} variants | {hardware}', flush=True)
    records = []
    pending = []
    for row in selected:
        done = verified_receipt(folder, row['reference_id'], signature_sha)
        if done is None:
            pending.append(row)
        else:
            records.append(done)
    print(f'{len(records)} sources already complete; {len(pending)} remaining.', flush=True)
    for name in ('records','native_inputs','reconstruction_eta0','variants','classifier_tiles'):
        (folder/name).mkdir(exist_ok=True)
    if pending:
        model, diffusion = create_model_and_diffusion(**(model_and_diffusion_defaults() | config))
        if diffusion.num_timesteps != 1000 or list(diffusion.timestep_map) != list(range(1000)):
            raise ValueError('Unexpected DDIM step schedule.')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True), strict=True)
        model.requires_grad_(False).eval().to(device)
        model.convert_to_fp16()
        def model_fn(x, t, y):
            if y.shape != (1,) or y.item() != 10 or torch.is_grad_enabled():
                raise RuntimeError('Both diffusion paths must use frozen, gradient-free H&E inference.')
            return model(x, t, y=y)
        timings = []
        target = torch.tensor([10], dtype=torch.long, device=device)
        with torch.inference_mode():
            for index, row in enumerate(pending):
                started = time.monotonic()
                ref = row['reference_id']
                repeat = row['repeat_checks'] == 'True'
                print(f'Starting {ref} | source {len(records)+1}/{len(selected)} | repeat checks {repeat}', flush=True)
                x = core.make_tensor(args.refs/row['image_path'], center_crop_arr, torch, device)
                native = (x[0].permute(1,2,0).cpu().numpy()*127.5+127.5).round().astype(np.uint8)
                files = {}
                def save_png(rel, image):
                    Image.fromarray(image).save(folder/rel)
                    files[rel] = core.digest(folder/rel)
                save_png(f'native_inputs/{ref}.png', native)
                latent = diffusion.ddim_reverse_sample_loop(model_fn, x, clip_denoised=False, eta=0.0,
                    model_kwargs={'y':target}, cond_fn=None, device=device, progress=True)
                if not latent.isfinite().all():
                    raise ValueError(f'{ref}: nonfinite inversion.')
                seeds = [seed_for(plan['seed'], ref, 'variant', i) for i in (1,2)]
                control_tensor = core.decode(diffusion, model_fn, latent, target, 0.0, seeds[0], torch, device)
                if repeat:
                    again = core.decode(diffusion, model_fn, latent, target, 0.0, seeds[1], torch, device)
                    if not torch.equal(control_tensor, again):
                        raise ValueError(f'{ref}: eta-zero reconstruction depends on its seed.')
                    del again
                control = core.to_image(control_tensor)
                save_png(f'reconstruction_eta0/{ref}.png', control)
                outputs = []
                pixels = []
                for i, seed in enumerate(seeds, 1):
                    decoded = core.decode(diffusion, model_fn, latent, target, 0.3, seed, torch, device)
                    if repeat and i == 1:
                        again = core.decode(diffusion, model_fn, latent, target, 0.3, seed, torch, device)
                        if not torch.equal(decoded, again):
                            raise ValueError(f'{ref}: same-seed stochastic decoding is not reproducible.')
                        del again
                    rgb = core.to_image(decoded)
                    pixels.append(rgb)
                    rel = f'variants/{ref}-{i:02d}.png'
                    classifier = f'classifier_tiles/{ref}-{i:02d}.png'
                    save_png(rel, rgb)
                    resized = np.asarray(Image.fromarray(rgb).resize((224,224), Image.Resampling.BICUBIC))
                    save_png(classifier, resized)
                    outputs.append(dict(**row, held_out_fold=args.fold, variant_id=f'{ref}-{i:02d}',
                        output_path=rel, output_sha256=files[rel], classifier_path=classifier,
                        classifier_sha256=files[classifier], reconstruction_path=f'reconstruction_eta0/{ref}.png',
                        output_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        classifier_pixel_sha256=hashlib.sha256(resized.tobytes()).hexdigest(),
                        reconstruction_sha256=files[f'reconstruction_eta0/{ref}.png'],
                        eta_encode=0, eta_decode=0.3, seed=seed, ddim_steps=1000,
                        source_domain='HE', target_domain='HE', model_input_px=128,
                        classifier_px=224, classifier_resize='128 to 224 bicubic',
                        ssim_vs_native_input=float(structural_similarity(native, rgb, channel_axis=2, data_range=255)),
                        ssim_vs_reconstruction=float(structural_similarity(control, rgb, channel_axis=2, data_range=255)),
                        checkpoint_sha256=weights['sha256'], repo_commit=core.COMMIT,
                        review_status='pending', training_eligible=False))
                if np.array_equal(pixels[0],pixels[1]):
                    raise ValueError(f'{ref}: distinct seeds produced identical image pixels.')
                elapsed = time.monotonic()-started
                receipt = dict(reference_id=ref, signature_sha256=signature_sha, outputs=outputs,
                    file_sha256=files, seconds=elapsed, passes=6 if repeat else 4,
                    reconstruction_ssim=float(structural_similarity(native,control,channel_axis=2,data_range=255)),
                    completed_at=core.stamp(), hardware=hardware, review_status='pending',training_eligible=False)
                atomic_json(folder/'records'/f'{ref}.json',receipt)
                records.append(receipt)
                timings.append((elapsed, receipt['passes']))
                recent = timings[-12:]
                per_pass=sum(t for t,p in recent)/sum(p for t,p in recent)
                left=pending[index+1:]
                remaining_passes=sum(6 if r['repeat_checks']=='True' else 4 for r in left)
                progress=dict(completed_sources=len(records), total_sources=len(selected), completed_variants=2*len(records),
                    remaining_sources=len(left), seconds_per_pass=per_pass,
                    estimated_seconds_remaining=remaining_passes*per_pass, updated_at=core.stamp(),
                    hardware=hardware, status='running' if left else 'finishing')
                atomic_json(folder/'progress.json',progress)
                print(f'COMPLETED {len(records)}/{len(selected)} sources | {2*len(records)} variants | '
                      f'{per_pass:.1f} s/pass | ETA {timedelta(seconds=round(progress["estimated_seconds_remaining"]))}',flush=True)
                del x, latent, control_tensor, decoded
    # Only publish a finished manifest after checking all expected source receipts.
    checked=[verified_receipt(folder,r['reference_id'],signature_sha) for r in selected]
    if any(r is None for r in checked):
        raise ValueError('A source receipt is missing at finalization.')
    rows=[row for receipt in checked for row in receipt['outputs']]
    core.save_csv(folder/'generation_manifest.csv',rows)
    core.require_hash(checkpoint,weights['sha256'])
    atomic_json(folder/'completed.json',dict(phase='batch',sources=len(selected),outputs=len(rows),
        signature_sha256=signature_sha,manifest_sha256=core.digest(folder/'generation_manifest.csv'),
        completed_at=core.stamp(),review_status='pending',training_eligible=False))
    print(f'Completed shard {args.shard}: {len(rows)} variants in {folder}',flush=True)


def status(args):
    plan, rows = validate_batch(args.batch)
    total=0
    for shard in (0,1):
        folder=args.batch/f'generated/shard-{shard}'
        receipts=list((folder/'records').glob('ref*.json'))
        total+=2*len(receipts)
        print(f'Shard {shard}: {len(receipts)}/{len(rows)//2} sources, {2*len(receipts)} variants')
        progress=folder/'progress.json'
        if progress.exists():
            info=core.read_json(progress)
            remaining=info['estimated_seconds_remaining']
            updated=datetime.fromisoformat(info['updated_at'])
            age=(datetime.now(timezone.utc)-updated).total_seconds()
            print(f'  Last measured {info["seconds_per_pass"]:.1f} s/pass; last ETA {timedelta(seconds=round(remaining))}; update age {age:.0f}s')
            print('  '+info['hardware'])
        print('  Finished' if (folder/'completed.json').exists() else '  Not finalized; inspect the shard log if progress stops')
    print(f'Total: {total}/{plan["synthetic_count"]} variants; review remains pending')


def verify(args):
    plan, refs=validate_batch(args.batch)
    reference_by_id=core.unique_by(refs,'reference_id')
    outputs=[]
    expected=list()
    for shard in (0,1):
        folder=args.batch/f'generated/shard-{shard}'
        done=core.read_json(folder/'completed.json')
        run=core.read_json(folder/'batch_run.json')
        core.require_hash(folder/'generation_manifest.csv',done['manifest_sha256'])
        selected=[r for r in refs if r['workload_shard']==str(shard)]
        checked=[verified_receipt(folder,r['reference_id'],run['signature_sha256']) for r in selected]
        if any(r is None for r in checked) or done['outputs']!=2*len(selected):
            raise ValueError('Missing completed sources or incorrect shard output count.')
        for receipt in checked:
            for row in receipt['outputs']:
                if row['patch_id']!=reference_by_id[receipt['reference_id']]['patch_id']:
                    raise ValueError('Output lineage differs from its source.')
                pref=f'generated/shard-{shard}/'
                outputs.append(dict(row, output_path=pref+row['output_path'],
                    classifier_path=pref+row['classifier_path'],reconstruction_path=pref+row['reconstruction_path']))
    if Counter(r['mgmt_patient_label'] for r in outputs)!=Counter(dict.fromkeys(LABELS,plan['sources_per_class']*2)):
        raise ValueError('Final synthetic label counts differ from the plan.')
    core.save_csv(args.batch/'generation_manifest.csv',outputs)
    atomic_json(args.batch/'completed.json',dict(outputs=len(outputs), verified_at=core.stamp(),
        manifest_sha256=core.digest(args.batch/'generation_manifest.csv'),review_status='pending',training_eligible=False))
    print(f'Verified {len(outputs)} variants, balanced by source-patient MGMT label. Review is pending.')


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare')
    for name in ('repo','root','package','labels','out'):s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--fold',type=int,choices=range(5),default=0)
    s.add_argument('--subset',choices=['A','B','full'],default='A')
    s.add_argument('--sources-per-class',type=int,default=1000)
    s.add_argument('--verify-per-class',type=int,default=8)
    s.add_argument('--seed',type=int,default=42)
    s.set_defaults(action=prepare)
    s=sub.add_parser('worker')
    for name in ('batch','upstream','weights'):s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--shard',type=int,choices=(0,1),required=True)
    s.add_argument('--device',default='cuda:0')
    s.set_defaults(action=worker)
    for name,action in [('status',status),('verify',verify)]:
        s=sub.add_parser(name);s.add_argument('--batch',type=Path,required=True);s.set_defaults(action=action)
    args=p.parse_args()
    if args.command=='prepare' and (not 2<=args.sources_per_class<=1000 or args.sources_per_class%2 or args.verify_per_class<1 or args.seed<0):
        p.error('Use an even 2–1000 sources per class, positive repeat-check count and nonnegative seed.')
    if args.command=='worker' and not core.re.fullmatch(r'cuda:\d+',args.device):p.error('Use a CUDA device, e.g. cuda:0')
    args.action(args)


if __name__=='__main__':
    try:cli()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f'Batch stopped: {error}')
