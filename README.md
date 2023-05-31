# MUAM
This project is used to segment the main body of paleontological fossil images.
Run MY_train.py directly for training.
Run MY_test.py directly for testing.
Details of experimental parameters are in the folder /configs/, and the parameters during training are MY_train.yml and during testing are MY.yml.
## Data structure
<pre><code>
MUAM
├── datasets
│   ├── DUTS
│   │   ├── Train
│   │   │   ├── images
│   │   │   ├── GT
│   │   │   ├── edges
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── DUT-O
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── GT
│   ├── HKU-IS
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── GT
      .
      .
      .
</code></pre>
## Requirements
* Python >= 3.7.x
* Pytorch >= 1.8.0
* albumentations >= 0.5.1
* tqdm >=4.54.0
* scikit-learn >= 0.23.2

## Run
<pre><code>
# For training
python MY_train.py

# For testing
python MY_test.py
</code></pre>
