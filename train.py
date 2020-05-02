# 训练模型
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from data_loader import get_loader
from vqa_models import VqaModel


### 根据实际环境调整设备 ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(params):
    # 生成日志目录与模型目录
    os.makedirs(params['log_dir'], exist_ok=True)
    os.makedirs(params['model_dir'], exist_ok=True)

    # 得到分批次的训练与验证数据集
    data_loader = get_loader(
        input_dir = params['input_dir'],
        input_vqa_train = 'train.npy',
        input_vqa_valid = 'valid.npy',
        max_qst_length = params['max_qst_length'],
        max_num_ans = params['max_num_ans'],
        batch_size = params['batch_size'],
        num_workers = params['num_workers']
        )

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size # 问题词典的长度(VqaDataset-VocabDict)
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size # 有效的答案的总长度
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx # 有效答案中未知词的索引

    #导入模型
    model = VqaModel(
        embed_size = params['embed_size'],
        qst_vocab_size = qst_vocab_size,
        ans_vocab_size = ans_vocab_size,
        word_embed_size = params['word_embed_size'],
        num_layers = params['num_layers'],
        hidden_size = params['hidden_size']
        ).to(device)

    criterion = nn.CrossEntropyLoss() # 多分类问题使用交叉熵损失
    # 罗列全部需要训练的参数
    parameters = \
        list(model.img_encoder.fc.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    # 设定用于更新参数的优化器
    optimizer = optim.Adam(parameters, lr=params['learning_rate'])

    # 设置调整学习率的机制
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])

    print('start training...')
    for epoch in range(params['num_epochs']): # 对于每次迭代
        for phase in ['train', 'valid']: # 分别运算训练样本与验证样本
            running_loss = 0.0 # 统计本次迭代中的交叉熵损失和
            running_corr_exp1 = 0 # 统计本次迭代中，预测值命中的有效答案数目
            running_corr_exp2 = 0 # 不计预测值命中<unk>的情况
            # 总共的batch数目
            batch_step_size = len(data_loader[phase].dataset) / params['batch_size']

            if phase == 'train': # 训练集的话，训练模型
                model.train()
            else:
                model.eval() # 验证集用来评估

            # 对于每个batch len(data_loader['train'])=2845
            for batch_idx, batch_sample in enumerate(data_loader[phase]):
                image = batch_sample['image'].to(device) # 4维数组
                question = batch_sample['question'].to(device) # 2维数组
                label = batch_sample['answer_label'].to(device) # batch_size*单标签
                multi_choice = batch_sample['answer_multi_choice'] # not tensor, list.

                optimizer.zero_grad() # 先将梯度置0
                # 只在训练时对梯度信息进行记录
                with torch.set_grad_enabled(phase=='train'):
                    # 代入数据得到输出值
                    output = model(image, question)      # [batch_size, ans_vocab_size=1000]
                    # 得到最大值所在的索引，即答案标签
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss = criterion(output, label) # 计算损失

                    if phase == 'train': # 训练集的话，根据损失更新参数、调整学习率
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                # 将预测为<unk>的标签值（0）设置为-9999
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item() # 累加每个batch的损失
                # 串联各样本结果[batch_size,10].求和看这些样本的有效答案（单个样本的有效答案是有重复的）中出现了多少个预测答案
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                # <unk> 命中不算
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                # 打印batch中的损失
                if batch_idx % 1 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, params['num_epochs'], batch_idx, int(batch_step_size), loss.item()))
                if batch_idx == 20: # too slow !!
                    break

            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            # 打印两种精度。（分母表示全部样本数，其实这个比例有点问题）
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset) # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset) # multiple choice
            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
                  .format(phase.upper(), epoch+1, params['num_epochs'], epoch_loss, epoch_acc_exp1, epoch_acc_exp2))

            # Log the loss and accuracy in an epoch.
            #保存本次迭代的batch平均损失、2种精度。（.item()用于取元素）
            with open(os.path.join(params['log_dir'], '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(
                    str(epoch+1) + '\t'
                    + str(epoch_loss) + '\t'
                    + str(epoch_acc_exp1.item()) + '\t'
                    + str(epoch_acc_exp2.item()))

        # Save the model check points.
        #训练集、验证集结束后，若迭代达到保存步，保存模型状态参数
        if (epoch+1) % params['save_step'] == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       os.path.join(params['model_dir'], 'model-epoch-{:02d}.ckpt'.format(epoch+1)))


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. The length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=156,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()
    params = vars(args)
    '''
    params = {
        'input_dir':'./datasets',
        'log_dir':'./logs',
        'model_dir':'./models',
        'max_qst_length':30,
        'max_num_ans':10,
        'embed_size':1024,
        'word_embed_size':300,
        'num_layers':2,
        'hidden_size':512,
        'learning_rate':0.01,
        'step_size':10,
        'gamma':0.1,
        'num_epochs':6,
        'batch_size':156,
        'num_workers':0, # 多线程
        # if num_workers = 8, for MacOS does not support CUDA,
        # IDLE may RAISE ERROR `The program is still running. Do you want to kill it?`
        'save_step':1
        }
    main(params)
