"""
Videoconferencing plugin demo for Linux.
v4l2loopback-utils needs to be installed, and a virtual webcam needs to be running at `--camera-device` (default: /dev/video1).
A target image and background should be supplied (default: demo_image.png and demo_video.mp4)


Once launched, the script is in background collection mode. Exit the frame and click to collect a background frame.
Upon returning, cycle through different target backgrounds by clicking.
Press Q any time to exit.

Example:
python demo_webcam.py --model-checkpoint "PATH_TO_CHECKPOINT" --resolution 1280 720 --hide-fps
"""

import argparse, os, shutil, time
import numpy as np
import cv2
import torch

from dataclasses import dataclass
from torch import nn
from torch.jit import ScriptModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
from threading import Thread, Lock
from tqdm import tqdm
from PIL import Image
import pyfakewebcam # pip install pyfakewebcam

# --------------- App setup ---------------
app = {
    "mode": "background",
    "bgr": None,
    "bgr_blur": None,
    "compose_mode": "plain",
    "target_background_frame": 0
}

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Virtual webcam demo')

parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=False)
parser.add_argument('--model-checkpoint-dir', type=str, required=False)

parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)

parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
parser.add_argument('--target-video', type=str, default='./demo_video.mp4')
parser.add_argument('--target-image', type=str, default='./demo_image.jpg')
parser.add_argument('--camera-device', type=str, default='/dev/video1')
parser.add_argument('--source_device_id', type=int, default=0)
args = parser.parse_args()


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'));
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


# An FPS tracker that computes exponentialy moving average FPS
class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.
class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        self.webcam = None
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)

    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        if self.webcam is not None:
            image_web = np.ascontiguousarray(image, dtype=np.uint8) # .copy()
            image_web = cv2.cvtColor(image_web, cv2.COLOR_RGB2BGR)
            self.webcam.schedule_frame(image_web)
        # else:
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF


class Controller: # A cv2 window with a couple buttons for background capture and cycling through target background options

    def __init__(self, model_checkpoint_dir):
        self.name = "RTHRBM Control"
        self.controls = [
            {
                "type": "button",
                "name": "mode_switch",
                "label": "Grab background",
                "x": 50,
                "y": 20,
                "w": 300,
                "h": 40
            },
            {
                "type": "button",
                "name": "compose_switch",
                "label": "Compose: plain white",
                "x": 50,
                "y": 80,
                "w": 300,
                "h": 40
            },
            {
                "type": "label",
                "name": "backbone_scale_label",
                "label": "Scale:  ",
                "x": 50,
                "y": 140,
                "w": 115,
                "h": 40
            },
            {
                "type": "label",
                "name": "backbone_scale_value",
                "label": str(bgmModel.backbone_scale),
                "x": 175,
                "y": 140,
                "w": 50,
                "h": 40
            },
            {
                "type": "button",
                "name": "backbone_scale_-",
                "label": "-",
                "x": 240,
                "y": 140,
                "w": 50,
                "h": 40
            },
            {
                "type": "button",
                "name": "backbone_scale_+",
                "label": "+",
                "x": 300,
                "y": 140,
                "w": 50,
                "h": 40
            },
            {
                "type": "label",
                "name": "refine_mode_label",
                "label": "Refine Mode",
                "x": 50,
                "y": 200,
                "w": 145,
                "h": 40
            },
            {
                "type": "button",
                "name": "refine_mode",
                "label": "",
                "x": 205,
                "y": 200,
                "w": 145,
                "h": 40
            },
            {
                "type": "label",
                "name": "refine_mode_label",
                "label": "",
                "x": 50,
                "y": 260,
                "w": 90,
                "h": 40
            },
            {
                "type": "label",
                "name": "refine_mode_value",
                "label": "",
                "x": 150,
                "y": 260,
                "w": 75,
                "h": 40
            },
            {
                "type": "button",
                "name": "refine_mode_-",
                "label": "-",
                "x": 240,
                "y": 260,
                "w": 50,
                "h": 40
            },
            {
                "type": "button",
                "name": "refine_mode_+",
                "label": "+",
                "x": 300,
                "y": 260,
                "w": 50,
                "h": 40
            },
            {
                "type": "button",
                "name": "model_checkpoint",
                "label": "Model: ",
                "x": 50,
                "y": 260,
                "w": 300,
                "h": 40
            }
        ]

        self.compose_cycle = [("plain", "Compose: plain white"), ("gaussian", "Compose: blur background"), ("video", "Compose: Winter holidays"), ("image", "Compose: Mt. Rainier")]
        self.refine_mode_cycle = [SamplingUpdater(self, 0), ThresholdUpdater(self, 1), FullUpdater(self, 2)]
        self.refine_mode_updater = self.refine_mode_cycle[self.getRefineModeIndex()]
        self.control_index = {}
        for control in self.controls:
            self.control_index[control["name"]] = control
        self.model_checkpoint_index = -1
        self.model_checkpoint_cycle = self.getModelCheckpointCycle(model_checkpoint_dir)
        self.refine_mode_updater.activate()
        cv2.namedWindow(self.name)
        cv2.moveWindow(self.name, 200, 200)
        cv2.setMouseCallback(self.name, self._raw_process_click)
        self.render()

    def getModelCheckpointCycle(self, model_checkpoint_dir):
        if model_checkpoint_dir is None:
            self.control_index["model_checkpoint"]["hidden"] = True
            return ()
        models = []
        current_model = self.to_short_model_name(bgmModel.model_checkpoint)
        with os.scandir(model_checkpoint_dir) as entries:
            for entry in entries:
                if 'torchscript_' in entry.name:
                    model_name = self.to_short_model_name(entry.name)
                    models.append((model_name, model_checkpoint_dir + entry.name))
                    print(entry.name)
                    if model_name == current_model:
                        self.model_checkpoint_index = len(models)
                        self.control_index["model_checkpoint"]["label"] = model_name
        return models

    def to_short_model_name(self, full_name):
        return full_name.split('torchscript_')[1]

    def getRefineModeIndex(self):
        for idx, updater in enumerate(self.refine_mode_cycle):
            if bgmModel.refine_mode == updater.name():
                return idx
        return -1
       
    def render(self):
        control_image = np.zeros((400,400, 3), np.uint8)
        for control in self.controls:
            if control.get("hidden") == True: continue
            if control["type"] == "button":
                bgColor = 180
            else:
                bgColor = (255, 255, 255)
            control_image[control["y"]:control["y"] + control["h"], control["x"]:control["x"] + control["w"]] = bgColor
            cv2.putText(control_image, control["label"], (control["x"] + 10, control["y"] + control["h"] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(self.name, control_image)
        
    def clicked(self, control):
        print("control clicked: " + control["name"])
        if control["name"] == "mode_switch":
            if app["mode"] == "background":
                grab_bgr()
                app["mode"] = "stream"
                control["label"] = "Select another background"
            else:
                app["mode"] = "background"
                control["label"] = "Grab background"
        elif control["name"] == "compose_switch":
            current_idx = next(i for i, v in enumerate(self.compose_cycle) if v[0] == app["compose_mode"])
            next_idx = (current_idx + 1) % len(self.compose_cycle)
            app["compose_mode"] = self.compose_cycle[next_idx][0]
            control["label"] = self.compose_cycle[next_idx][1]
        elif control["name"] == "backbone_scale_+":
            self.updateBackboneScale(round(min(bgmModel.backbone_scale + .10, 1.0), 2))
        elif control["name"] == "backbone_scale_-":
            self.updateBackboneScale(round(max(bgmModel.backbone_scale - .10, 0.0), 2))
        elif control["name"] == "refine_mode":
            next_index = (self.refine_mode_updater.index + 1) % len(self.refine_mode_cycle)
            self.refine_mode_updater = self.refine_mode_cycle[next_index]
            self.refine_mode_updater.activate()
        elif control["name"] == "refine_mode_-":
            self.refine_mode_updater.updateModelValue(-1)
        elif control["name"] == "refine_mode_+":
            self.refine_mode_updater.updateModelValue(1)
        elif control["name"] == "model_checkpoint":
            self.updateModelCheckpoint()
        self.render()

    def updateBackboneScale(self, scale):
        bgmModel.backbone_scale = scale
        self.control_index["backbone_scale_value"]["label"] = str(scale) 
        bgmModel.reload()

    def updateModelCheckpoint(self):
        next_idx = (self.model_checkpoint_index + 1) % len(self.model_checkpoint_cycle)
        bgmModel.model_checkpoint = self.model_checkpoint_cycle[next_idx][1]
        print(self.model_checkpoint_cycle[next_idx][1])
        self.model_checkpoint_index = next_idx
        self.control_index["model_checkpoint"]["label"] = self.model_checkpoint_cycle[next_idx][0]
        bgmModel.reload()

    def _raw_process_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for control in self.controls:
                if control.get("hidden") == True: continue
                if x > control["x"] and x < control["x"] + control["w"] and y > control["y"] and y < control["y"] + control["h"]:
                    self.clicked(control)


class RefineModeUpdater:

    def __init__(self, controller, index):
        self.controller = controller
        self.index = index

    def index(self): pass
    def name(self): pass
    def activate(self): pass
    def updateModelValue(self, delta): pass

    def setControlsHidden(self, isHidden):
        self.controller.control_index["refine_mode_label"]["hidden"] = isHidden
        self.controller.control_index["refine_mode_value"]["hidden"] = isHidden
        self.controller.control_index["refine_mode_-"]["hidden"] = isHidden
        self.controller.control_index["refine_mode_+"]["hidden"] = isHidden


class SamplingUpdater(RefineModeUpdater):

    def name(self):
        return 'sampling'

    def activate(self):
        bgmModel.refine_mode = self.name()
        bgmModel.refine_sample_pixels = 80000
        self.controller.control_index["refine_mode"]["label"] = self.name()
        self.controller.control_index["refine_mode_label"]["label"] = 'pixels'
        self.updateValueLabel()
        self.setControlsHidden(False)

    def updateValueLabel(self):
        self.controller.control_index["refine_mode_value"]["label"] = str(bgmModel.refine_sample_pixels)
        bgmModel.reload()

    def updateModelValue(self, delta):
        bgmModel.refine_sample_pixels = max(bgmModel.refine_sample_pixels + delta * 40000, 0)
        self.updateValueLabel()


class ThresholdUpdater(RefineModeUpdater):

    def name(self):
        return 'thresholding'

    def activate(self):
        bgmModel.refine_mode = self.name()
        bgmModel.refine_threshold = .1
        self.controller.control_index["refine_mode"]["label"] = self.name()
        self.controller.control_index["refine_mode_label"]["label"] = 'threshold'
        self.updateValueLabel()
        self.setControlsHidden(False)

    def updateValueLabel(self):
        self.controller.control_index["refine_mode_value"]["label"] = str(bgmModel.refine_threshold)
        bgmModel.reload()

    def updateModelValue(self, delta):
        bgmModel.refine_threshold = min(max(bgmModel.refine_threshold + delta * .01, 0),1.0)
        self.updateValueLabel()


class FullUpdater(RefineModeUpdater):

    def name(self):
        return 'full'

    def activate(self):
        bgmModel.refine_mode = self.name()
        self.controller.control_index["refine_mode"]["label"] = self.name()
        bgmModel.reload()
        self.setControlsHidden(True)

    def updateModelValue(self, delta):
        # nothing to do
        return


class VideoDataset(Dataset):
    def __init__(self, path: str, transforms: any = None):
        self.cap = cv2.VideoCapture(path)
        self.transforms = transforms
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.cap.read()
        if not ret:
            raise IndexError(f'Idx: {idx} out of length: {len(self)}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()

@dataclass
class BGModel:
    model_checkpoint: str
    backbone_scale: float
    refine_mode: str
    refine_sample_pixels: int
    refine_threshold: float

    def model(self):
        return self.model
        
    def reload(self):
        self.model = torch.jit.load(self.model_checkpoint)
        self.model.backbone_scale = self.backbone_scale
        self.model.refine_mode = self.refine_mode
        self.model.refine_sample_pixels = self.refine_sample_pixels
        self.model.model_refine_threshold = self.refine_threshold
        print(self.refine_mode + " , " + str(self.refine_sample_pixels) + ", " + str(self.refine_threshold))
        self.model.cuda().eval()        
        

# --------------- Main ---------------


# Load model

bgmModel = BGModel(args.model_checkpoint, args.model_backbone_scale, args.model_refine_mode, args.model_refine_sample_pixels, args.model_refine_threshold)
#bgmModel.reload() # Controller kicks it off.

source_device_id = args.source_device_id
width, height = args.resolution
cam = Camera(device_id=source_device_id, width=width, height=height)
dsp = Displayer('RTHRBM Preview', cam.width, cam.height, show_info=(not args.hide_fps))
ctr = Controller(args.model_checkpoint_dir)
fake_camera = pyfakewebcam.FakeWebcam(args.camera_device, cam.width, cam.height)
dsp.webcam = fake_camera

def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()

preloaded_image = cv2_frame_to_cuda(cv2.imread(args.target_image))
tb_video = VideoDataset(args.target_video, transforms=ToTensor())


def grab_bgr():
    bgr_frame = cam.read()
    bgr_blur = cv2.GaussianBlur(bgr_frame.astype('float32'), (67, 67), 0).astype('uint8') # cv2.blur(bgr_frame, (10, 10))
    app["bgr"] = cv2_frame_to_cuda(bgr_frame)
    app["bgr_blur"] = cv2_frame_to_cuda(bgr_blur)


def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img


def hologram_effect(img):
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 2, 3

    for y in range(holo.shape[0]):
        if y % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)

    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)

    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out

def frame_to_hologram(pha, fgr):
    mask = pha * fgr
    mask_img = mask.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
    mask_img = hologram_effect(mask_img)
    return cv2_frame_to_cuda(mask_img) * pha


def app_step():
    if app["mode"] == "background":
        frame = cam.read()
        key = dsp.step(frame)
        if key == ord('q'):
            return True
    else:
        frame = cam.read()
        src = cv2_frame_to_cuda(frame)
        pha, fgr = bgmModel.model(src, app["bgr"])[:2]
        if app["compose_mode"] == "plain":
            tgt_bgr = torch.ones_like(fgr)
        elif app["compose_mode"] == "image":
            tgt_bgr = nn.functional.interpolate(preloaded_image, (fgr.shape[2:]))
        elif app["compose_mode"] == "video":
            vidframe = tb_video[app["target_background_frame"]].unsqueeze_(0).cuda()
            tgt_bgr = nn.functional.interpolate(vidframe, (fgr.shape[2:]))
            app["target_background_frame"] += 1
        elif app["compose_mode"] == "gaussian":
            tgt_bgr = app["bgr_blur"]

        hologram_mask = frame_to_hologram(pha, fgr)
        res = hologram_mask + (1 - pha) * tgt_bgr
        res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        #res = pha * fgr
        #res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        #res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        #res = hologram_effect(res)
        #src = cv2_frame_to_cuda(res) * pha

        #res = pha * fgr + (1 - pha) * tgt_bgr
        #res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        #res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        ghiDK8ZhiEqWNCEGBvsiwAVg

        key = dsp.step(res)

        if key == ord('q'):
            return True

with torch.no_grad():
    while True:
        if app_step():
            break
