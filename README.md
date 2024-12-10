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

    Layer (type)                  ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 180, 180, 8)    │           608 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 90, 90, 8)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 90, 90, 8)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 90, 90, 16)     │         1,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 45, 45, 16)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 45, 45, 16)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 45, 45, 32)     │        12,832 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 22, 22, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 22, 22, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 15488)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     3,965,184 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 12)             │         3,084 │

1. **`conv2d` (Conv2D):** Extracts 8 feature maps from the input image using 2D convolution with a kernel, reducing spatial complexity.  
2. **`max_pooling2d` (MaxPooling2D):** Down-samples the feature maps by taking the maximum value in a 2x2 window, reducing spatial dimensions.  
3. **`dropout`:** Randomly drops some connections to reduce overfitting during training.  
4. **`conv2d_1` (Conv2D):** Extracts 16 higher-level feature maps from the pooled output, adding complexity to the learned features.  
5. **`max_pooling2d_1`:** Further reduces spatial dimensions while retaining key features.  
6. **`dropout_1`:** Adds regularization to prevent overfitting in this layer.  
7. **`conv2d_2` (Conv2D):** Extracts 32 more detailed feature maps, capturing more complex patterns.  
8. **`max_pooling2d_2`:** Further reduces the size of feature maps to make computation efficient.  
9. **`dropout_2`:** Applies regularization again to prevent overfitting.  
10. **`flatten`:** Flattens the 3D feature maps into a 1D vector for input to the dense layers.  
11. **`dense`:** A fully connected layer with 256 neurons to learn patterns and relationships in the flattened features.  
12. **`dropout_3`:** Regularizes the fully connected layer to avoid overfitting.  
13. **`dense_1`:** Output layer with 12 neurons, corresponding to the 12 classification steps, using softmax activation.

## Results



# phys323-final
