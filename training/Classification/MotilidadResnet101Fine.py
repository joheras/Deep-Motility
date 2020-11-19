from fastai.vision.all import *
import fastai

torch.cuda.set_device(0)

path="dataset"

bs=10
size=512

data=ImageDataLoaders.from_folder(path,batch_tfms=aug_transforms(),item_tfms=Resize(size),bs=bs)

save=SaveModelCallback(monitor='accuracy',fname='model-MotilidadResNet101Fine-512')
learn=cnn_learner(data,resnet101,metrics=[accuracy,F1Score(),Precision(),Recall()],cbs=[save])
model=learn.load('model-MotilidadResNet101').model

learn=Learner(data,model,metrics=[accuracy,F1Score(),Precision(),Recall()],cbs=[save])


learn.fine_tune(15,freeze_epochs=2)
learn.save('MotilidadResnet101Fine-512')

