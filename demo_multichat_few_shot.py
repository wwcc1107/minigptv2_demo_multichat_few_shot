import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, Conversation

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='minigptv2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1, help="specify the gpu to load the model.")
    parser.add_argument("--temperature", type=int, default=0.6, help="specify the gpu to load the model.")
    parser.add_argument("--prompt", type=str, default="/nfs/volume-512-1/wangchang/MiniGPT-4-2/multi_chat/1.txt", help="第一个few shot样本的query，这里将query写到了txt文本当中，便于读取")
    parser.add_argument("--answer-txt", type=str, default="/nfs/volume-512-1/wangchang/MiniGPT-4-2/multi_chat/answer_84_2.txt", help="将文件夹内的图像批量测试的推理结果写入到这个txt文本文件当中")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================
CONV_VISION_minigptv2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you, Please answer my questions.",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')
nums = 0
while True:
    image_path = "/nfs/dataset-libstdm_situation_awareness/wangchang/testcase_17_69/img_guiji_1/858140270.png"
    # 第一张few-shot图像
    if image_path == 'stop':
        break
    if len(image_path) > 0:
        query = args.prompt

    chat_state = CONV_VISION_minigptv2.copy()
    img_list = []
    img_list2 = []
    chat.upload_img(image_path, chat_state, img_list)
    img_list2.append(img_list)
    txt_path = args.answer_txt
    
    while True:
        if nums == 90:
            sys.exit(0)
            # nums == 90的时候，标志着文件夹的图像已经批量推理完成
        if  90 > nums >= 6:
            # 一共是6个few shot的案例样本，所以nums >= 6,
            # 需要批量测试的图像共有84张，所以nums <= 84+6
            query = "question.txt"
            # 需要批量测试的文件夹内的图像的prompt，写入到一个txt文本文件中，方便读取
            image_path = "/img/"
            # 需要批量测试的文件夹内的图像路径
            imgs = glob.glob(os.path.join(image_path, '*')) 
            image_path = imgs[nums-6]
            img_list = []
            chat_state = chat_state
            chat.upload_img(image_path, chat_state, img_list)
            img_list2.append(img_list)
        if nums == 1:
            query = "2.txt"
            # 第2个few shot样本的query，这里将query写到了txt文本当中，便于读取
            image_path = "/8582632.png"
            # 第2个few shot样本的图像
            image_path = image_path
            img_list = []
            chat_state = chat_state
            chat.upload_img(image_path, chat_state, img_list)
            img_list2.append(img_list)
        if nums == 2:
            query = "3.txt"
            # 第3个few shot样本的query，这里将query写到了txt文本当中，便于读取
            image_path = "/858263.png"
            # 第3个few shot样本的图像
            image_path = image_path
            img_list = []
            chat_state = chat_state
            chat.upload_img(image_path, chat_state, img_list)
            img_list2.append(img_list)
        if nums == 3:
            query = "4.txt"
            # 第4个few shot样本的query，这里将query写到了txt文本当中，便于读取
            image_path = "/85826.png"
            # 第4个few shot样本的图像
            image_path = image_path
            img_list = []
            chat_state = chat_state
            chat.upload_img(image_path, chat_state, img_list)
            img_list2.append(img_list)
        if nums == 4:
            query = "5.txt"
            # 第5个few shot样本的query，这里将query写到了txt文本当中，便于读取
            image_path = "/8582.png"
            # 第5个few shot样本的图像
            image_path = image_path
            img_list = []
            chat_state = chat_state
            chat.upload_img(image_path, chat_state, img_list)
            img_list2.append(img_list)
        if nums == 5:
            query = "6.txt"
            # 第6个few shot样本的query，这里将query写到了txt文本当中，便于读取
            image_path = "/85.png"
            # 第6个few shot样本的图像
            image_path = image_path
            img_list = []
            chat_state = chat_state
            chat.upload_img(image_path, chat_state, img_list)
            img_list2.append(img_list)
            # 以上一共是6个few shot的案例样本

        image_path = image_path
        img_list = img_list
        chat_state = chat_state
        f3 = open(query, 'r', encoding='utf-8')
        lines = f3.read() 
        query = str(lines)
        chat.ask(query, chat_state)
        chat.encode_img(img_list2[-1])

        if nums >= 6:
            llm_message = chat.answer(
                conv=chat_state,
                img_list=img_list2[-1],
                num_beams=args.num_beams,
                temperature=args.temperature,
                max_new_tokens=500,
                max_length=2000
            )[0]
            f1 = open(txt_path, 'a')
            x = image_path.split("/")[-1]
            f1.write(x + ':' + llm_message + '\t')
            f1.write('\n')
        nums += 1
