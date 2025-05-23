import torch
from transformers import AutoTokenizer
from distrl.models.critic import VLMDoubleCritic, TrajectoryCritic
from .model import T5ForMultimodalGeneration
import signal

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from collections import deque
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class TARSAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = None):
        super(TARSAgent, self).__init__()

        policy_lm = "/mnt/huangke1/model_TARS/ByteDance-Seed/UI-TARS-1.5-7B"  
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                     "/mnt/huangke1/model_TARS/ByteDance-Seed/UI-TARS-1.5-7B",
                          torch_dtype=torch.bfloat16,
                           #    attn_implementation="flash_attention_2",
        #                        device_map="auto",
                                 ).to(device)
        self.processor = AutoProcessor.from_pretrained("/mnt/huangke1/model_TARS/ByteDance-Seed/UI-TARS-1.5-7B")
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.critic = VLMDoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.target_critic = None
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
        self.answer = ""
        self.msg_q = deque(maxlen=15)
        self.MOBILE_USE = """you are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='xxx')

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""     
        self.messages = []
    
    def prepare(self):
        print("to be define....")
        #self.model = self.accelerator.prepare(self.model)


    def get_action(self, observation, image_features):
        image_path = observation[0]["image_path"]
        task = observation[0]["task"]
        
        messages_example2 ={}
        messages_example2["role"]="user"
        messages_example2["content"]=[]
        tmp_im={}
        tmp_im["type"]="image"
        tmp_im["image"]=image_path
        messages_example2["content"].append(tmp_im)
        tmp_te={}
        tmp_te["type"]="text"
        tmp_te["text"]=self.MOBILE_USE  +"\nUser Instruction: " + task
        messages_example2["content"].append(tmp_te)

    
        #self.messages.append(messages_example2)
        self.msg_q.append(messages_example2)

        print("length ..multi turn....==",(len(list(self.msg_q))+1)/2)
        for _ in range(3):
            try:
                with timeout(seconds=120):
                    with torch.no_grad():
                        print("promnt>>>>>>>>>>>>>>>>>>>>")
                        #print(self.messages)
                        print(list(self.msg_q))
                        text = self.processor.apply_chat_template(list(self.msg_q), tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(list(self.msg_q))
                        inputs = self.processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
                        inputs = inputs.to(self.device)

                        text_token_count = inputs["input_ids"].shape[1]  # 形状为 [batch_size, seq_len]

                        #visual_token_per_image = 256  # 具体取决于模型配置

                        image_token_count = len(image_inputs) *10000
                        total_token_count = text_token_count + image_token_count 
                        print("tttttkkkkk,text_token_count   ",text_token_count)

                        generated_ids = self.accelerator.unwrap_model(self.model).generate(**inputs, max_new_tokens=128).cpu()
                        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        print(output_text)
                        raw_action = output_text[0].split("\nAction: ")[-1]
                        print(raw_action)
                        mess_ans={}
                        mess_ans["role"]="assistant"
                        mess_ans["content"]=output_text[0]
                        #self.messages.append(mess_ans)
                        self.msg_q.append(mess_ans)
                        raw_action = [raw_action]
                    break
            except TimeoutError:
                print("Timeout while accessing actions")
                continue
        #raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        #for _ in range(3):
            #raw_action = [a[1:] if a.startswith('\n') for a in raw_action]
        # return raw_action
        return raw_action

    def get_log_prob(self, observation, image_features, action):
        image_features = image_features[...,-1408:]
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        image_features = image_features.to(self.device)
        outputs = self.accelerator.unwrap_model(self.model)(
            input_ids = obs_ids["input_ids"],
            image_ids = image_features,
            attention_mask = obs_ids["attention_mask"],
            labels = action_ids["input_ids"]
        )
        
        # # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        # input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        # attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
        #                         dim = 1)
        # outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        # values = None
        # if isinstance(outputs, Tuple):
        #     values, outputs = outputs
        ## TODO: need to check if token shifting is done correctly
        prediction_probs = self.softmax(outputs.logits)
        selected_prediction_probs = torch.take_along_dim(prediction_probs,\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        selected_prediction_probs = torch.clamp(selected_prediction_probs, min=0.001, max=0.99)
        # import IPython; IPython.embed(); exit()
        return torch.log(selected_prediction_probs)*action_ids["attention_mask"]
