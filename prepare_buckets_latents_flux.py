import argparse
import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from library.device_utils import get_preferred_device, init_ipex

init_ipex()

from torchvision import transforms

import library.train_util as train_util
from library.flux_utils import load_ae
from library.strategy_flux import FluxLatentsCachingStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEVICE = get_preferred_device()

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_npz_filename(data_dir, image_key, is_full_path, recursive, resolution=None):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    # Add resolution and architecture to filename if resolution is provided
    if resolution:
        base_name = f"{base_name}_{resolution[0]}x{resolution[1]}_flux"

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name) + ".safetensors"
    else:
        return os.path.join(data_dir, base_name) + ".safetensors"

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB mode, handling transparency if needed.
    """
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def load_and_convert_to_pil(image_path):
    """
    Load an image from disk, convert it to PIL, ensure RGB format, and return the image and path.
    """
    try:
        image = Image.open(image_path)
        image = pil_ensure_rgb(image)
        return (image, image_path)
    except Exception as e:
        logger.error(f"Error loading or converting image {image_path}: {e}")
        return None

def main(args):
    flux_strategy = FluxLatentsCachingStrategy(
        cache_to_disk=True, batch_size=args.batch_size, skip_disk_cache_validity_check=True
    )
    # assert args.bucket_reso_steps % 8 == 0, f"bucket_reso_steps must be divisible by 8 / bucket_reso_stepは8で割り切れる必要があります"
    if args.bucket_reso_steps % 8 > 0:
        logger.warning(
            "resolution of buckets in training time is a multiple of 8 / 学習時の各bucketの解像度は8単位になります"
        )
    if args.bucket_reso_steps % 32 > 0:
        logger.warning(
            "WARNING: bucket_reso_steps is not divisible by 32. It is not working with SDXL / bucket_reso_stepsが32で割り��れません。SDXLでは動作しません"
        )

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, args.recursive)]
    logger.info(f"found {len(image_paths)} images.")

    if os.path.exists(args.in_json):
        logger.info(f"loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.error(f"no metadata / メタデータファイルがありません: {args.in_json}")
        #return
        metadata = json.loads("{}")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = load_ae(args.model_name_or_path, weight_dtype, DEVICE)

    # bucketのサイズを計算する
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert (
        len(max_reso) == 2
    ), f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

    bucket_manager = train_util.BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )
    if not args.bucket_no_upscale:
        bucket_manager.make_buckets()
    else:
        logger.warning(
            "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
        )

    # 画像をひとつずつ適切なbucketに割り当てながらlatentを計算する
    img_ar_errors = []

    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                flux_strategy.cache_batch_latents(vae, bucket, args.flip_aug, args.alpha_mask, False)
                bucket.clear()

    # Parallel conversion using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        converted_images = list(executor.map(load_and_convert_to_pil, image_paths))

    bucket_counts = {}

    for i, result in enumerate(converted_images):
        if result is None:
            continue

        image, image_path = result
        image_key = image_path if args.full_path else os.path.splitext(os.path.basename(image_path))[0]
        metadata[image_key] = {}

        # Bucket assignment and processing
        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        metadata[image_key]["train_resolution"] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)

        if not args.bucket_no_upscale:
            assert (
                    resized_size[0] == reso[0] or resized_size[1] == reso[1]
            ), f"Internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
            assert (
                    resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
            ), f"Internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

        assert (
                resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
        ), f"Internal error, resized size is small: {resized_size}, {reso}"

        # Get original image dimensions for the filename
        original_size = (image.width, image.height)
        npz_file_name = get_npz_filename(args.train_data_dir, image_key, args.full_path, args.recursive, original_size)

        # Add to batch
        image_info = train_util.ImageInfo(image_key, 1, False, image_path)
        image_info.latents_cache_path = npz_file_name
        image_info.bucket_reso = reso
        image_info.resized_size = resized_size
        image_info.image = np.array(image)
        bucket_manager.add_image(reso, image_info)

        image.close()
        del image
        converted_images[i] = None
        if i % 30 == 0:
            gc.collect()

        # Decide whether to process the batch
        process_batch(False)

    # 残りを処理する
    process_batch(True)

    bucket_manager.sort()
    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            logger.info(f"bucket {i} {reso}: {count}")
    img_ar_errors = np.array(img_ar_errors)
    logger.info(f"mean ar error: {np.mean(img_ar_errors)}")

    # metadataを書き出して終わり
    logger.info(f"writing metadata: {args.out_json}")
    with open(args.out_json, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument(
        "model_name_or_path", type=str, help="model name or path to encode latents / latentを取得するためのモデル"
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="not used (for backward compatibility) / 使用されません（互換性のため残してあります）",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="512,512",
        help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）",
    )
    parser.add_argument(
        "--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度"
    )
    parser.add_argument(
        "--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最大解像度"
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習��像ディレクトリに対応）",
    )
    parser.add_argument(
        "--flip_aug",
        action="store_true",
        help="flip augmentation, save latents for flipped images / 左右反転した画像もlatentを取得、保存する",
    )
    parser.add_argument(
        "--alpha_mask",
        type=str,
        default="",
        help="save alpha mask for images for loss calculation / 損失計算用に画像のアルファマスクを保存する",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
