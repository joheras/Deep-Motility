from fastai.vision.all import *
import fastai
from wwf.vision.timm import *

torch.cuda.set_device(0)

path="dataset"

bs=8
size=512

dls=ImageDataLoaders.from_folder(path,batch_tfms=aug_transforms(),item_tfms=Resize(size),bs=bs)

save=SaveModelCallback(monitor='accuracy',fname='model-MotilidadEfficient-512')
#learn=cnn_learner(data,resnet50,metrics=[accuracy,F1Score(),Precision(),Recall()],cbs=[save])
learn = timm_learner(dls, 'efficientnet_b3a', metrics=[accuracy,F1Score(),Precision(),Recall()],cbs=[save])


learn.fine_tune(15,freeze_epochs=2)
learn.save('MotilidadEfficient-512')

