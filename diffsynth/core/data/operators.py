import json
import os
import random

import imageio
import imageio.v3 as iio
import torch
import torchvision

from PIL import Image, ImageOps


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value

    def __call__(self, data):
        if data is None:
            data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB

    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB:
            image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        random_start: bool = False,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.random_start = random_start

    def _adjust_num_frames(self, total_frames: int | None):
        """
        Adjust requested num_frames to fit within total_frames and satisfy
        time_division_factor/time_division_remainder constraints.
        """
        num_frames = self.num_frames
        if total_frames is not None and total_frames < num_frames:
            num_frames = int(total_frames)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        # Be explicit about the backend to reduce ambiguity across environments.
        # This still relies on ffmpeg availability (imageio[ffmpeg] / imageio-ffmpeg).
        reader = imageio.get_reader(data, "ffmpeg")
        try:
            try:
                total_frames = int(reader.count_frames())
            except Exception:
                # Some containers/codecs don't support reliable frame counting.
                total_frames = None

            num_frames = self._adjust_num_frames(total_frames)
            if total_frames is not None and self.random_start and total_frames > num_frames:
                start_frame = random.randint(0, total_frames - num_frames)
            else:
                start_frame = 0

            frames = []
            for frame_id in range(start_frame, start_frame + num_frames):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            return frames
        except Exception as e:
            # Add high-signal context; upstream will decide whether to skip or crash.
            raise RuntimeError(
                f"Failed to decode video with imageio/ffmpeg: path={data!r}, "
                f"requested_num_frames={self.num_frames}, "
                f"time_division_factor={self.time_division_factor}, "
                f"time_division_remainder={self.time_division_remainder}, "
                f"random_start={self.random_start}."
            ) from e
        finally:
            try:
                reader.close()
            except Exception:
                pass


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator

    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        random_start: bool = False,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.random_start = random_start

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        if self.random_start and len(images) > num_frames:
            start_frame = random.randint(0, len(images) - num_frames)
            images = images[start_frame : start_frame + num_frames]
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, data: str):
        import librosa

        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


# Wan I2V default: 832x480, aspect 832/480
WAN_I2V_WIDTH = 832
WAN_I2V_HEIGHT = 480
WAN_I2V_ASPECT = WAN_I2V_WIDTH / WAN_I2V_HEIGHT


def _pad_to_wan_ratio_and_resize(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Pad image to Wan aspect ratio (width/height = 832/480), then resize to target size."""
    width, height = image.size
    target_width_by_ratio = int(round(height * WAN_I2V_ASPECT))
    if width < target_width_by_ratio:
        pad_total = target_width_by_ratio - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        image = ImageOps.expand(image, border=(pad_left, 0, pad_right, 0), fill=0)
        width = target_width_by_ratio
    elif width > target_width_by_ratio:
        # Crop width to ratio
        crop_total = width - target_width_by_ratio
        crop_left = crop_total // 2
        image = image.crop((crop_left, 0, crop_left + target_width_by_ratio, height))
        width = target_width_by_ratio
    return image.resize((target_width, target_height), Image.Resampling.BILINEAR)


class LoadBlenderFlowersFrames(DataProcessingOperator):
    """
    Load a sequence of frames from Blender flowers dataset renders/ with random start,
    frame step, and optional random view. Pads to Wan I2V aspect ratio and resizes to
    target size. Expects directory layout: <base_path>/<renders_subdir>/<scene_name>_<frame_idx:03d>/mv.png
    (or other view images / subdirs for random view).
    """

    def __init__(
        self,
        base_path: str,
        renders_subdir: str = "renders",
        num_frames: int = 49,
        frame_step: int = 2,
        random_start: bool = True,
        random_view: bool = True,
        target_height: int = WAN_I2V_HEIGHT,
        target_width: int = WAN_I2V_WIDTH,
        convert_RGB: bool = True,
    ):
        self.base_path = base_path
        self.renders_subdir = renders_subdir
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.random_start = random_start
        self.random_view = random_view
        self.target_height = target_height
        self.target_width = target_width
        self.convert_RGB = convert_RGB

    def _list_frame_indices(self, scene_name: str):
        renders_dir = os.path.join(self.base_path, self.renders_subdir)
        if not os.path.isdir(renders_dir):
            return []
        prefix = scene_name + "_"
        indices = []
        for name in os.listdir(renders_dir):
            if not name.startswith(prefix):
                continue
            path = os.path.join(renders_dir, name)
            if not os.path.isdir(path):
                continue
            suffix = name[len(prefix) :]
            if len(suffix) == 3 and suffix.isdigit():
                indices.append(int(suffix))
        return sorted(indices)

    def _load_transforms_frames(self, frame_dir: str) -> list | None:
        """Load transforms.json in frame_dir; return list of frame entries (views) or None."""
        path = os.path.join(frame_dir, "transforms.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            frames = data.get("frames")
            if isinstance(frames, list) and len(frames) > 0:
                return frames
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _get_frame_image_path(self, frame_dir: str, view_idx: int | None) -> str:
        """Return path to one image in frame_dir. If view_idx is not None, use transforms.json
        and frames[view_idx]['file_path']; otherwise fall back to mv.png or directory scan."""
        if view_idx is not None:
            frames = self._load_transforms_frames(frame_dir)
            if frames is not None and view_idx < len(frames):
                file_path = frames[view_idx].get("file_path")
                if file_path:
                    full = os.path.join(frame_dir, file_path)
                    if os.path.isfile(full):
                        return full
        # Fallback: mv.png or first available image/view
        candidates = []
        mv_path = os.path.join(frame_dir, "mv.png")
        if os.path.isfile(mv_path):
            candidates.append(mv_path)
        for name in os.listdir(frame_dir):
            subpath = os.path.join(frame_dir, name)
            if os.path.isfile(subpath) and name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                if subpath not in candidates:
                    candidates.append(subpath)
            elif os.path.isdir(subpath):
                view_mv = os.path.join(subpath, "mv.png")
                if os.path.isfile(view_mv):
                    candidates.append(view_mv)
        if not candidates:
            raise FileNotFoundError(f"No image found in frame dir: {frame_dir}")
        return random.choice(candidates) if self.random_view else candidates[0]

    def __call__(self, data: str):
        raw = data.strip()
        # Support reverse direction: video column can be "scene_name|reverse" (prompt = reverse growing)
        if raw.endswith("|reverse"):
            scene_name = raw[:-8].strip()
            reverse = True
        else:
            scene_name = raw
            reverse = False

        indices = self._list_frame_indices(scene_name)
        if not indices:
            raise RuntimeError(f"Scene {scene_name} has no frames in renders.")

        need = (self.num_frames - 1) * self.frame_step + 1  # e.g. 97 for 49 frames step 2

        if len(indices) >= need:
            # Enough frames: random start (if enabled), then every frame_step-th frame
            if self.random_start:
                max_start = len(indices) - need
                start_idx_pos = random.randint(0, max_start)
            else:
                start_idx_pos = 0
            selected_indices = [
                indices[start_idx_pos + i * self.frame_step]
                for i in range(self.num_frames)
            ]
        else:
            # Fewer than need (e.g. daisy 80 frames): always start from first frame, step by frame_step, pad by duplicating last frame
            start_idx_pos = 0
            selected_indices = []
            for i in range(self.num_frames):
                pos = start_idx_pos + i * self.frame_step
                if pos < len(indices):
                    selected_indices.append(indices[pos])
                else:
                    selected_indices.append(indices[-1])
            assert len(selected_indices) == self.num_frames

        renders_dir = os.path.join(self.base_path, self.renders_subdir)
        frame_dirs = [
            os.path.join(renders_dir, f"{scene_name}_{idx:03d}")
            for idx in selected_indices
        ]
        # Pick one view index from transforms.json valid for all frames in this sequence
        view_idx = None
        if self.random_view:
            min_views = None
            for fd in frame_dirs:
                view_frames = self._load_transforms_frames(fd)
                n = len(view_frames) if view_frames else 0
                min_views = n if min_views is None else min(min_views, n)
            if min_views is not None and min_views > 0:
                view_idx = random.randint(0, min_views - 1)
        frames = []
        for frame_dir in frame_dirs:
            img_path = self._get_frame_image_path(frame_dir, view_idx)
            image = Image.open(img_path)
            if self.convert_RGB:
                image = image.convert("RGB")
            image = _pad_to_wan_ratio_and_resize(
                image, self.target_width, self.target_height
            )
            frames.append(image)
        if reverse:
            frames = list(reversed(frames))
        return frames
