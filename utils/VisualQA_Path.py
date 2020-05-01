from os.path import join

root = '/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/VisualQA'

anno = join(root, 'annotations')
train_anno = join(anno, 'v2_mscoco_train2014_annotations.json')
val_anno = join(anno, 'v2_mscoco_val2014_annotations.json')

ques = join(root, 'questions')
train_ques = join(ques, 'v2_OpenEnded_mscoco_train2014_questions.json')
val_ques = join(ques, 'v2_OpenEnded_mscoco_val2014_questions.json')
test_dev_ques = join(ques, 'v2_OpenEnded_mscoco_test-dev2015_questions.json')
test_ques = join(ques, 'v2_OpenEnded_mscoco_test2015_questions.json')

img = join(root, 'images')
train_img_path = join(img, 'train2014') # + '/COCO_train2014_000000001401.jpg'
val_img_path = join(img, 'val2014') # + '/COCO_val2014_000000000164.jpg'
test_img_path = join(img, 'test2015') # + '/COCO_test2015_000000000057.jpg'

pairs = join(root, 'complementary_pairs')
train_pairs = join(pairs, 'v2_mscoco_train2014_complementary_pairs.json')
val_pairs = join(pairs, 'v2_mscoco_val2014_complementary_pairs.json')

if __name__ == '__main__':
    print(f'train_anno = {train_anno}')
    print(f'val_anno = {val_anno}')
    print(f'train_ques = {train_ques}')
    print(f'val_ques = {val_ques}')
    print(f'test_dev_ques = {test_dev_ques}')
    print(f'test_ques = {test_ques}')
    print(f'train_img_path = {train_img_path}')
    print(f'val_img_path = {val_img_path}')
    print(f'test_img_path = {test_img_path}')
    print(f'train_pairs = {train_pairs}')
    print(f'val_pairs = {val_pairs}')
