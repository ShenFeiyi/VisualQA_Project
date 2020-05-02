# 整合数据
import numpy as np
import json
import os
import argparse

from text_helper import tokenize, VocabDict
from VisualQA_Path import root


def extract_answers(q_answers, valid_answer_set):
    """
    :param q_answers:        10个回答字典组成得到回答列表
    :param valid_answer_set: 根据词频选择的top答案
    :return:                 全部的回答词语列表，在top中的答案形成的有效答案列表
    """
    all_answers = [ answer["answer"] for answer in q_answers ]
    valid_answers = [ a for a in all_answers if a in valid_answer_set ]
    return all_answers, valid_answers

def vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, image_set):
    if image_set in ['train2014', 'val2014']: # 存在答案
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations']
            '''
            a list
            annotations[i] = {
            'question_type',
            'multiple_choice_answer',
            'answers': [
                {'answer',
                'answer_confidence',
                'answer_id'},...,{}
                ]
            'image_id',
            'answer_type',
            'question_id'
            }
            '''
            # 建立问题编号与解释字典间的字典
            qid2ann_dict = { ann['question_id']:ann for ann in annotations }
            print(f'annotations in {annotation_file.split("/")[-1:][0] % image_set} loaded.')
    else:
        load_answer = False

    with open(question_file % image_set) as f:
        questions = json.load(f)['questions'] #列表。每个元素为字典，对应每个Image_Q
        '''
        a list
        questions[i] = {
        'image_id',
        'question',
        'question_id'
        }
        '''
        print(f'questions of {question_file.split("/")[-1:][0] % image_set} loaded.')
    coco_set_name = image_set.replace('-dev', '')
    # get all questions in variable `question` & all annotations in `annotations`

    # 绝对路径
    abs_image_dir = os.path.abspath(image_dir % coco_set_name) # test-dev2015的Questions仍对应图像文件夹test2015
    image_name_template = 'COCO_' + coco_set_name + '_%012d' # 图像名字模板
    dataset = [None]*len(questions) # 全部问题数目
    
    unk_ans_count = 0 # 没有有效回答的例子数目
    for n_q, q in enumerate(questions):#对于每一问题
        if (n_q+1) % 10000 == 0:
            print(f'processing {round(10000*(n_q+1)/len(questions))/100}%')
        image_id = q['image_id'] # 与问题对应的图像id
        question_id = q['question_id'] # 问题id
        image_name = image_name_template % image_id # 图像文件名
        image_path = os.path.join(abs_image_dir, image_name+'.jpg') # 图像路径 # all in .jpg format
        question_str = q['question'] # 问题文本
        question_tokens = tokenize(question_str) # 返回句子切词列表

        #构建信息字典
        iminfo = dict(
            image_name = image_name,
            image_path = image_path,
            question_id = question_id,
            question_str = question_str,
            question_tokens = question_tokens
            )
        
        if load_answer: # 如果有回答
            ann = qid2ann_dict[question_id] # 通过问题id对应到解释字典
            # 抽取回答集合
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)

            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1 # 没有有效回答的例子数目+1

            iminfo['all_answers'] = all_answers # 往信息表中添加答案信息
            iminfo['valid_answers'] = valid_answers
            
        dataset[n_q] = iminfo # 根据问题文件序列存储信息
    print(f'in {image_set}: total {unk_ans_count} out of {len(questions)} answers are <unk>\n')
    return dataset # 返回整合后的数据集，列表型，元素为字典


def main(params):
    # 设定图像路径、答案路径、问题路径
    image_dir = params['output_dir']+'/images/%s/'
    annotation_file = params['input_dir']+'/annotations/v2_mscoco_%s_annotations.json'
    question_file = params['input_dir']+'/questions/v2_OpenEnded_mscoco_%s_questions.json'

    vocab_answer_file = params['output_dir']+'/annotations/vocab_answers.txt'
    answer_dict = VocabDict(vocab_answer_file) # 建立类
    valid_answer_set = set(answer_dict.word_list) # 载入文件得到单词列表并将其作为有效的回答集合

    # 分别对4类文件夹做处理
    train = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2014')
    valid = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'val2014')
    test = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test2015')
    test_dev = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test-dev2015')

    # 保存整合后的数据集到npy文件
    if not os.path.exists(params['output_dir']+'/preprocessed_data'):
        os.makedirs(params['output_dir']+'/preprocessed_data')
    np.save(params['output_dir']+'/preprocessed_data/train.npy', np.array(train))
    np.save(params['output_dir']+'/preprocessed_data/valid.npy', np.array(valid))
    np.save(params['output_dir']+'/preprocessed_data/train_valid.npy', np.array(train+valid))
    np.save(params['output_dir']+'/preprocessed_data/test.npy', np.array(test))
    np.save(params['output_dir']+'/preprocessed_data/test-dev.npy', np.array(test_dev))


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/VisualQA',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='../datasets',
                        help='directory for outputs')
    
    args = parser.parse_args()
    params = vars(args)
    main(params)
    '''
    params = {
        'input_dir':root, # ~/VisualQA
        'output_dir':'../datasets'
        }
    main(params)
