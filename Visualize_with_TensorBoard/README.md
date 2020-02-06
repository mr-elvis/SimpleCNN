# **Simple Visualization of Machine Learning model with TensorBoard**

### **What is TensorBoard**?

[TensorBoard](https://www.tensorflow.org/tensorboard) provides the visualization and tooling needed for machine learning experimentation. it's very useful for monitoring of model’s training process. 

In this example, we are going to use TensorBoard to visualize and track metrics such as Loss and accuracy of a model.

### **How to use TensorBoard**

1. *Import TensorBoard* 

   ```python
   from tensorflow.python.keras.callbacks import TensorBoard
   import time
   ```

2. *After the model definition instantiate TensorBoard and specify a log directory where events will be saved for visualization.*

   ```python
   tensorboard = Tensorboard('log_dir/{}'.format(time.time()))
   ```

3. In the fit function, tell Keras to call back to the TensorBoard by specifying the Callback parameter using the instance of  the TensorBoard.

   ```python
   model.fit(x_train,y_train, epochs=10, callbacks=[tensorboard])
   ```

4. In your Terminal execute the tensorboard command pointing to the log directory you specified. It will execute and gives a localhost address, copy and access the address in a browser to visualize  the training process. You can also view the graph that was built for the training.

   ```python
   >>tensorboard --logdir="specify_path_to_your_log_directory"
   ```

   





