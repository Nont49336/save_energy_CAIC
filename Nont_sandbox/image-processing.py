
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import load_img

image_size1 = 240
image_size2 = 480
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size1, image_size2, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
from keras.models import Sequential

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 5
val_batchsize = 2

train_dir = './images/train'
validation_dir = './images/validate'
print(train_dir)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size1, image_size2),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size1, image_size2),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

# Save the model
model.save('image_recmodel.h5')


#graph ploting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size1, image_size2),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator,
                                      steps=validation_generator.samples / validation_generator.batch_size, verbose=1)

predicted_classes = np.argmax(predictions, axis=1)




#start of error identify

#errors = np.where(predicted_classes != ground_truth)[0]
#print("No of errors = {}/{}".format(len(errors), validation_generator.samples))
correct = np.where(predicted_classes == ground_truth)[0]
print("No of correct = {}/{}".format(len(correct),validation_generator.samples))

# Show the errors picture
# for i in range(len(errors)):
#     pred_class = np.argmax(predictions[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         predictions[errors[i]][pred_class])
#     original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
#     plt.figure(figsize=[7, 7])
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(original)
#     plt.show()

for i in range(len(correct)):
    pred_class = np.argmax(predictions[correct[i]])
    print(pred_class)
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[correct[i]].split('/')[0],
        pred_label,
        predictions[correct[i]][pred_class])
    original = load_img('{}/{}'.format(validation_dir, fnames[correct[i]]))
    plt.figure(figsize=[7, 7])
    plt.axis('off') #not importance
    plt.title(title)
    plt.imshow(original)
    plt.show()


#testing set
image = load_img('images/test/noon.jpg', target_size=(240, 480))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
pred = model.predict(image)
if(pred[0][0]>pred[0][1]):
    pred_label = "night"
else:
    pred_label ="noon"

title = ' Prediction :{}'.format(
    pred_label
)
image_show = load_img('images/test/noon.jpg', target_size=(240, 480))
plt.figure(figsize=[7, 7])
plt.axis('off')
plt.title(title)
plt.imshow(image_show)
plt.show()

model.save('image_recmodel.h5')