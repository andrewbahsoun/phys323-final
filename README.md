# Physics 323 Final Project

## Project Context
Originally, I had a video data set containing videos of 12 different hand washing steps. Part of the dataset came from a public dataset, [available here](https://zenodo.org/records/4537209). Then, my GCI group recorded more videos (~30 minutes per step).    

From here, we submitted our project to the GCI program. We used a PyTorch model that got hand data points using mediaPipe and trained a simple model on that. This model was ok, but had errors and only a 60% accuracy rate. Its failures came from the mediaPipe module being unable to recognize hands that were on top of eachother.

We ended up winning the best GCI project. [News.](https://news.chapman.edu/2024/05/31/this-ai-hand-washing-coach-may-help-prevent-the-spread-of-deadly-viruses/) But I knew this project could be improved with a CNN Image Classification Model. 

Now, I have taken on this project again to develop an image classification model. I always wanted to do this project at some point but didn't have the time. I decided to take it on for my Physics 323 final project. Thank you to Dr. Dressel for giving his students freedom on their projects, it allowed me to complete this project.

## Data Preprocessing

In the jupyter notebook ``data_preprocessing.ipynb``, set input_dir to the directory containing videos of hand_washing. Be sure that they are seperated by step. So, input_dir should contain directories like 'step_1', 'step_2', ... , 'step_12'. 

``output_frames_dir_test`` is the location that the test frames will end up (30% of the frames, this can be changed by changing the ``test_ratio`` variable. Currently it is at 0.3). They will be organized by directory in steps as well.     

``output_frames_dir_train`` is the location that the train frames will end up. 

The preprocessing works by entering every video and taking screenshots of each frame using cv2. It saves each frame as a jpg image and adds it to the respective directory. 


## Training the model

The model is trained using a basic tensorflow CNN model. Its layers look like such: 

┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ sequential_2 (Sequential)            │ (None, 128, 128, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rescaling_3 (Rescaling)              │ (None, 128, 128, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 128, 128, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_4                │ (None, 128, 128, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 64, 64, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 64, 64, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_5                │ (None, 64, 64, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_4 (MaxPooling2D)       │ (None, 32, 32, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 32, 32, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_6                │ (None, 32, 32, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_5 (MaxPooling2D)       │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_1 (Flatten)                  │ (None, 32768)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 256)                 │       8,388,864 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_7                │ (None, 256)                 │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 12)                  │           3,084 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 8,487,116 (32.38 MB)
 Trainable params: 8,486,156 (32.37 MB)
 Non-trainable params: 960 (3.75 KB)

### Explaination of the Layers

1. **`Rescaling` (Rescaling):**
   - Normalizes input pixel values from [0, 255] to [0, 1] for consistency

2. **`Conv2D` (conv2d_3):**
   - Applies 32 convolutional filters of size 3*3 to extract low-level features like edges

3. **`BatchNormalization` (batch_normalization_4):**
   - Normalizes the feature maps to stabilize and speed up training.

4. **`MaxPooling2D` (max_pooling2d_3):**
   - Reduces the spatial dimensions to 64*64, retaining important features and reducing computation.

5. **`Conv2D` (conv2d_4):**
   - Applies 64 convolutional filters of size 3*3 to learn more complex patterns

6. **`BatchNormalization` (batch_normalization_5):**
   - Normalizes the feature maps for the second convolutional layer.

7. **`MaxPooling2D` (max_pooling2d_4):**
   - Further reduces spatial dimensions to 32*32, summarizing learned features

8. **`Conv2D` (conv2d_5):**
   - Uses 128 filters to capture high-level features at 32*32 resolution.

9. **`BatchNormalization` (batch_normalization_6):**
   - Normalizes the feature maps for the third convolutional layer.

10. **`MaxPooling2D` (max_pooling2d_5):**
    - Reduces spatial dimensions to 16*16, retaining essential information.

11. **`Dropout` (dropout_2):**
    - Randomly drops 50% of neurons to reduce overfitting during training.

12. **`Flatten` (flatten_1):**
    - Converts the 3D tensor 16*16*128 into a 1D vector of size 32,768 for dense layers.

13. **`Dense` (dense_2):**
    - Fully connected layer with 256 neurons to learn complex relationships from the flattened features.

14. **`BatchNormalization` (batch_normalization_7):**
    - Normalizes the outputs of the dense layer for stability and faster convergence.

15. **`Dropout` (dropout_3):**
    - Drops 50% of the neurons in the dense layer to prevent overfitting.

16. **`Dense` (dense_3):**
    - Final output layer with 12 neurons, each representing a class. Uses softmax activation to output class probabilities.

## Results



# phys323-final
