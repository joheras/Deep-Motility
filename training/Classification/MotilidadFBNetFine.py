from fastai.vision.all import *
import fastai
import sys
sys.path.insert(0, '../../ecai2020-adrian/melanoma/mobile_vision/')

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet



torch.cuda.set_device(0)

path="dataset"

bs=8
size=512

dls=ImageDataLoaders.from_folder(path,batch_tfms=aug_transforms(),item_tfms=Resize(size),bs=bs)


def create_squeezenet_body(cut=None):
  model = fbnet("dmasking_l3",pretrained=True)
  return model.backbone


body = create_squeezenet_body()

nf = num_features_model(nn.Sequential(*body.children())) * (2)
head = create_head(nf, dls.c)
model = nn.Sequential(body, head)

apply_init(model[1], nn.init.kaiming_normal_)


save=SaveModelCallback(monitor='accuracy',fname='model-MotilidadFBNetFine-512')
learn=Learner(dls,model,metrics=[accuracy,F1Score(),Precision(),Recall()],cbs=[save])
model=learn.load('model-MotilidadFBNet').model
learn=Learner(dls,model,metrics=[accuracy,F1Score(),Precision(),Recall()],cbs=[save])

learn.fine_tune(15,freeze_epochs=2)
learn.save('MotilidadFBNetFine-512')

