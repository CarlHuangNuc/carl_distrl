from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Union
from transformers import Blip2VisionModel, AutoProcessor, Blip2Model
import torch
from PIL import Image
from time import sleep
use_tars = True

from .ui_tars.action_parser import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS, MAX_RATIO, parse_action_to_structure_output, parsing_response_to_pyautogui_code
import re

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                width: int,
                factor: int = IMAGE_FACTOR,
                min_pixels: int = MIN_PIXELS,
                max_pixels: int = MAX_PIXELS,) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class ImageFeatureExtractor:
    def __init__(self, device):
        # Set device based on CUDA availability
        self.device = device
        
        # Initialize and load the BLIP2 model and processor
        self.model = Blip2Model.from_pretrained("/mnt/public/huangke1/auto-UI/blip2-opt-2.7b").cpu()
        self.model.language_model = None
        # self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained("/mnt/public/huangke1/auto-UI/blip2-opt-2.7b")

    def to_feat(self, image: Image.Image):
        """Converts a PIL image to a feature representation using the BLIP2 model.
        
        Args:
            image: A PIL.Image object representing the image to convert.
            
        Returns:
            A tensor representing the image feature.
        """
        with torch.no_grad():
            # Preprocess the image and move to the correct device
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            
            # Get the image features from the model
            image_features = self.model.get_image_features(**inputs).pooler_output[0]
            
            # Detach the tensor from the graph and move it to CPU
            image_features = image_features.detach().cpu()
            
        return image_features

class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7

@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None

    def __str__(self):
        # Construct the basic action type string.
        components = [f"Action Type: {self.action_type.name}"]

        # Format and add touch_point if it's not None.
        if self.touch_point:
            touch_point_str = f"({self.touch_point[0]:.4f}, {self.touch_point[1]:.4f})"
            components.append(f"Touch Point: {touch_point_str}")

        # Format and add lift_point if it's not None.
        if self.lift_point:
            lift_point_str = f"({self.lift_point[0]:.4f}, {self.lift_point[1]:.4f})"
            components.append(f"Lift Point: {lift_point_str}")

        # Add typed_text if it's not None.
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")

        # Join all components into a single string.
        return ", ".join(components)

    def to_act(self):
        pass


def autoui_translate_action(out):
    if not use_tars:
        action_str = out.split("Action Decision: ")[1]
        action_type, touch_point_1, touch_point_2, lift_point_1, lift_point_2, typed_text = action_str.split(", ")
        touch_point = touch_point_1 + ", " + touch_point_2
        lift_point = lift_point_1 + ", " + lift_point_2
        try:
            action_type = action_type.split(": ")[1].strip('"')
            if action_type == 'DUAL_POINT':
                touch_point_yx = touch_point.split(": ")[1].strip('[]"')
                touch_point_yx = [float(num) for num in touch_point_yx.split(", ")]
                lift_point_yx = lift_point.split(": ")[1].strip('[]"')
                lift_point_yx = [float(num) for num in lift_point_yx.split(", ")]
                return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point_yx[::-1], lift_point=lift_point_yx[::-1])
            elif action_type == 'TYPE':
                text = typed_text.split(": ")[1].strip('"')
                return AndroidAction(action_type=ActionType.Type, typed_text=text)
            elif action_type == 'PRESS_HOME':
                return AndroidAction(action_type=ActionType.GoHome)
            elif action_type == 'PRESS_BACK':
                return AndroidAction(action_type=ActionType.GoBack)
            elif action_type == 'PRESS_ENTER':
                return AndroidAction(action_type=ActionType.Enter)
            elif action_type == 'STATUS_TASK_COMPLETE':
                return AndroidAction(action_type=ActionType.TaskComplete)
            elif action_type == 'TASK_IMPOSSIBLE':
                return AndroidAction(action_type=ActionType.TaskImpossible)
            else:
                print(f"Action {out} not supported yet.")
                return AndroidAction(action_type=ActionType.Idle)
        except Exception as e:
            print(f"Action {out} Parsing Error: {e}")
            return AndroidAction(action_type=ActionType.Idle)
    else:
        original_image_width, original_image_height = 1080, 2280
        parsed_dict = parse_action_to_structure_output(out,factor=1000,
                    origin_resized_height=original_image_height,
                        origin_resized_width=original_image_width,
                            model_type="qwen25vl")

        print(parsed_dict)
       
        """
        parsed_pyautogui_code = parsing_response_to_pyautogui_code(
             responses=parsed_dict,
                 image_height=original_image_height,
                     image_width=original_image_width
                     )
        print(parsed_pyautogui_code)
        """
        ## for osworld sample interface 
        if parsed_dict[0]["action_type"] == "click":
            match = re.search(r'\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]', parsed_dict[0]["action_inputs"]["start_box"])
            if match:
                x1 = float(match.group(1))
                y1 = float(match.group(2))
                x2 = float(match.group(3))
                y2 = float(match.group(4))
                #print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            else:
                print("未找到匹配的坐标")
            nor_x = x1
            nor_y = y1
            print(nor_x)
            print(nor_y)
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=[nor_x,nor_y], lift_point=[nor_x,nor_y])
        elif parsed_dict[0]["action_type"] == "type":
            return AndroidAction(action_type=ActionType.Type, typed_text=parsed_dict[0]["action_inputs"]["content"])

        elif parsed_dict[0]["action_type"] == "wait":
            sleep(5)
            print("sleep ..........................")

            return AndroidAction(action_type=ActionType.Enter)
        else:
            print("unknown ..... dddddddddddddddddddddddd..................")
            exit()


def to_autoui(act: AndroidAction):
    if act.action_type == ActionType.DualPoint:
        return f'"action_type": "DUAL_POINT", "touch_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]", "lift_point": "[{act.lift_point[1]:.4f}, {act.lift_point[0]:.4f}]", "typed_text": ""'
    elif act.action_type == ActionType.Type:
        return f'"action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "{act.typed_text}"'
    elif act.action_type == ActionType.GoBack:
        return f'"action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.GoHome:
        return f'"action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.Enter:
        return f'"action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
        return f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    else:
        print(f"Action {act} not supported yet.")
        return ""

def autoui_prepare_prompt(task, history):
        prompt = "Previous Actions: "
        for act in history[-1:]:
            prompt += f"{to_autoui(act)} "
        prompt += f"Goal: {task}</s>"
        return prompt
