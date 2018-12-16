# lung-nodule-classification
This model actually helps radiologist to draw the region of interest and then come up with a conclusion whether the ROI is a nodule or not.
<br>
To get started with the model
<ul>
  <li> Open your command prompt</li>
  <li> change your path so that it is now in this directory, or just type cmd in address bar from this directory</li>
  <li> type 'python main.py', this will train the model for 100 epochs</li>
  <li> after the training has been done, type 'python inference.py' and it will load the default test.png, draw as many bounding box you need to draw in that image.</li>
  <li> Having done that, we need to feed those bounding boxes to the neural network, to do so, press 'q' on your keyboard</li>
  <li> You'll be presented with an image with the prediction</li>
 </ul>
