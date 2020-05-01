# 载入初始数据
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from utils import text_helper


# 映射型数据
class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/preprocessed_data'+'/'+input_vqa,allow_pickle=True) # 整合后的数据集
        self.qst_vocab = text_helper.VocabDict(input_dir+'/questions'+'/vocab_questions.txt') # 建立类
        self.ans_vocab = text_helper.VocabDict(input_dir+'/annotations'+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length # 设置问题长度
        self.max_num_ans = max_num_ans  # 设置答案数目
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None) # Ture or False. 有效答案没有的都赋成了['<unk>']
        self.transform = transform

    def __getitem__(self, idx): # 得到每一实例

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans

        image = vqa[idx]['image_path']
        image = Image.open(image).convert('RGB') # 转换为RGB形式
        # 初始化问题索引列表
        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length) # padded with '<pad>' in 'ans_vocab'
        # 根据实际问题情况修改索引
        qst2idc[:len(vqa[idx]['question_tokens'])] = [ qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens'] ]
        # 得到样本字典，图像3元数组，问题1维索引数组
        sample = {'image': image, 'question': qst2idc}

        if load_ans:
            # 得到有效答案的索引列表
            ans2idc = [ ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers'] ]
            ans2idx = np.random.choice(ans2idc) # 从数组中随机抽取一个元素
            # 给定样本标签。标签总长度为抽取的有效答案字典.txt长度
            sample['answer_label'] = ans2idx         # for training

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            # 给定多选形式的答案列表，非有效集中的答案标签赋为-1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

        if transform: # 转换图像数据
            sample['image'] = transform(sample['image'])

        return sample # 返回单样本字典

    def __len__(self):

        return len(self.vqa) # 返回数据集的大小


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):
     # 将图像数据转换为tensor型，并使用给定的均值标准差做归一化处理
     # 分别建立训练与验证的转换器
    transform = {
        phase: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
                )
            ]) for phase in ['train', 'valid']
        }

    # 分别建立训练与验证的数据载入类
    vqa_dataset = {
        'train': VqaDataset(
            input_dir = input_dir,
            input_vqa = input_vqa_train,
            max_qst_length = max_qst_length,
            max_num_ans = max_num_ans,
            transform = transform['train']
            ),
        'valid': VqaDataset(
            input_dir = input_dir,
            input_vqa = input_vqa_valid,
            max_qst_length = max_qst_length,
            max_num_ans = max_num_ans,
            transform = transform['valid']
            )
        }

    # 分别对训练与验证数据划分批次
    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset = vqa_dataset[phase], # 映射类型的数据集
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers
            ) for phase in ['train', 'valid']
        }

    return data_loader # 返回字典
