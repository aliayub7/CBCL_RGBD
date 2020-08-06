import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os

class Img2Vec():

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        #self.scaler = transforms.Scale((224, 224))
        #self.scaler = transforms.Resize((224, 224))
        self.resizer = transforms.Resize(256)
        self.scaler = transforms.CenterCrop(224)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])#(mean = [0.6983, 0.3918, 0.4474],std = [0.1648, 0.1359, 0.1644])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:

            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name == 'alexnet':
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name == 'alexnet':
                    return my_embedding.numpy()[:, :]
                else:
                    print(my_embedding.numpy()[:, :, 0, 0].shape)
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            #image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
            # applying transforms
            img = self.resizer(img)
            img = self.scaler(img)
            img = self.to_tensor(img)
            if len(img)==3:
                img = self.normalize(img)
                image = img.unsqueeze(0).to(self.device)
                #out = self.model(image)
                #ind = out.max(-1)[1]
                #print ('returned label',ind)

                if self.model_name == 'alexnet':
                    my_embedding = torch.zeros(1, self.layer_output_size)
                else:
                    my_embedding = torch.zeros(1, self.layer_output_size,1,1)

                def copy_data(m, i, o):
                    my_embedding.copy_(o.data)

                h = self.extraction_layer.register_forward_hook(copy_data)
                h_x = self.model(image)
                h.remove()

                if tensor:
                    return my_embedding
                else:
                    if self.model_name == 'alexnet':
                        return my_embedding.numpy()[0,: ,0,0]
                    else:
                        return my_embedding.numpy()[0, :,0,0]
                """
                features = self.model(image)
                features = features.cpu().detach().numpy()[0]
                return features
                """
                #return my_embedding
            else:
                return None

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet_places':
            #model = models.resnet18(pretrained=True)
            arch = 'resnet18'
            # load the pre-trained weights
            model_file = '%s_places365.pth.tar' % arch
            if not os.access(model_file, os.W_OK):
                weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                os.system('wget ' + weight_url)

            model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)

            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)
            return model, layer

        elif model_name == 'SUN_vgg':
            #model = models.resnet18()
            #model.fc = nn.Linear(in_features = 512, out_features = 19)
            model = models.vgg16()
            model.classifier[6] = nn.Linear(in_features = 4096, out_features = 15)
            model.load_state_dict(torch.load("./checkpoint/best_after200SUN_15classes"))
            if layer=='default':
                #layer = model.classifier[5]
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            model.classifier = new_classifier
            return model, layer
        elif model_name == 'NYU_vgg':
            model = models.vgg16()
            model.classifier[6] = nn.Linear(in_features = 4096, out_features = 10)
            model.load_state_dict(torch.load("./checkpoint/best_after200NYU"))
            if layer=='default':
                layer = model.classifier[5]
                self.layer_output_size = 4096
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            model.classifier = new_classifier
            return model, layer
        elif model_name == 'resnet-34':
            model = models.resnet34(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)
            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)
