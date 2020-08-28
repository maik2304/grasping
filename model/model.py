from torchvision import datasets, models, transforms
import torch.nn as nn
import torch

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224


    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class MyResNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyResNet, self).__init__()
        self.pretrained = my_pretrained_model
        '''      
        for param in self.pretrained.conv1.parameters():
            param.requires_grad = False
        for param in self.pretrained.layer1.parameters():
            param.requires_grad = False
        for param in self.pretrained.layer2.parameters():
            param.requires_grad = False  
        '''
        self.pretrained  = nn.Sequential(*list(self.pretrained.children())[:-1]) 

        self.my_new_layers = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(512, 6))
        
        self.my_new_layers.apply(init_weights)

    def forward(self, x):
        x = self.pretrained(x)
        x = x.view(-1,512)
        x = self.my_new_layers(x)
        return x
