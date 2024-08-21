'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import torch
from torch import nn
from opts import parse_opts
from datasets.ravdess import RAVDESS
import transforms 

from torch.autograd import Variable

from models import multimodalcnn

import warnings
warnings.filterwarnings("ignore")


def generate_model(opt):
    assert opt.model in ['multimodalcnn']

    if opt.model == 'multimodalcnn':   
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)


    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    
    return model, model.parameters()








'''def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'RAVDESS':
        training_data = RAVDESS(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    return training_data


if __name__ == '__main__':

    opt = parse_opts()
    model, parameters = generate_model(opt)
    print("Generated Model Successfully!")

    # Video Transform
    video_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotate(),
                    transforms.ToTensor(opt.video_norm_value)])
    

    training_data = get_training_set(opt, spatial_transform=video_transform) 
        
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True)
    
    print("Loaded Data using Data Loader!")


    # print(len(next(iter(train_loader))))
    for i, (audio_inputs, visual_inputs, targets) in enumerate(train_loader):
        print(f"i: {i}")
        print(f"audio_inputs: {audio_inputs.shape}")
        print(f"visual_inputs: {visual_inputs.shape}")
        print(f"targets: {targets.shape}")

        targets = targets.to(opt.device)

        with torch.no_grad():
                
            if opt.mask == 'noise':
                audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)                   
                visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0) 
                targets = torch.cat((targets, targets, targets), dim=0)                    
                shuffle = torch.randperm(audio_inputs.size()[0])
                audio_inputs = audio_inputs[shuffle]
                visual_inputs = visual_inputs[shuffle]
                targets = targets[shuffle]
                
            elif opt.mask == 'softhard':
                coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100
                vision_coefficients = 1 - coefficients
                coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
                vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

                audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0) 
                visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)   
                
                targets = torch.cat((targets, targets, targets, targets), dim=0)
                shuffle = torch.randperm(audio_inputs.size()[0])
                audio_inputs = audio_inputs[shuffle]
                visual_inputs = visual_inputs[shuffle]
                targets = targets[shuffle]

        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)

        targets = Variable(targets)
        print("Preprocessing Inputs Done!")
        outputs = model(audio_inputs, visual_inputs)

        print(outputs.shape)

        break


# print(model(IP2, IP1).shape)'''
