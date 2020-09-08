from torchvision import datasets, models, transforms
import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=1.0)
        m.bias.data.fill_(0.01)

class MyResNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyResNet, self).__init__()
        self.pretrained = my_pretrained_model
        
        # freeze the first convolutional layers
        '''      
        for param in self.pretrained.conv1.parameters():
            param.requires_grad = False
        for param in self.pretrained.layer1.parameters():
            param.requires_grad = False
        for param in self.pretrained.layer2.parameters():
            param.requires_grad = False  
        '''
        #self.pretrained = nn.Sequential(*list(self.pretrained.children())[:-1]) 
        
        # creation of the new fully connected layers
        my_new_layers = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(512, 6))
        
        # replace the fully connected layer of ResNet34 with the new layers
        self.pretrained.fc = my_new_layers
        
        # initialization of the new layers. Default: Gaussian initialization
        #self.my_new_layers.apply(init_weights)

    def forward(self, x):
        x = self.pretrained(x)        
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):   
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet34
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
    
    elif model_name == "myresnet":
        # load the pretrained network resnet34
        pretrained = models.resnet34(pretrained=True)
        model_ft = MyResNet(pretrained)
        
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft

