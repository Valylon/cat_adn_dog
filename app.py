import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import zipfile
import glob

files_zip_ext = glob.glob('/*.zip')
print(files_zip_ext)
def extract_data_from_zip(file_path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall("/data")

# Extract train and test
for file_path in files_zip_ext:
    extract_data_from_zip(file_path)


train_data = len(os.listdir('/train'))
test_data = len(os.listdir('/test1'))


@app.route('/gn', methods=['GET'])
def step3():
    ros.chdir('/data')
    train_y = (lambda dir_: [1 if file.split('.')[0] == 'dog' else 0 for file in os.listdir(dir_)])('train')

gen_path = lambda dir_: [path for path in os.listdir(dir_)]
train_x = gen_path('train')
test_x = gen_path('test1')train_y = (lambda dir_: [1 if file.split('.')[0] == 'dog' else 0 for file in os.listdir(dir_)])('train')

gen_path = lambda dir_: [path for path in os.listdir(dir_)]
train_x = gen_path('train')
test_x = gen_path('test1')
df = pd.DataFrame({'filename': train_x,
                    'category': train_y})
return df.tail().sns.displot(df, x='category')


@app.route('/viz', methods=['GET'])
def visualize(img_path):
    img = mpimg.imread(img_path)

 plt.figure(figsize=(8,8))
 plt.imshow(img)
 visualize(f"train/{df['filename'].iloc[0]}")



@app.route('/preprocess', methods=['GET'])
def preprocess(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # resize the images
    size = 28
    gray = cv2.resize(gray, (size, size))

    # normalize
    normalized = gray.flatten() / 255.0

    # global centering
    mean = normalized.mean()
    centered = normalized - mean

    return centered.reshape(1, size*size)

@app.route('/gb', methods=['GET'])
def gen_batches(X, y=None, batch_size=200, image_size=784):
    batch = []
    for i, x in enumerate(X, start=1):
        img = preprocess(x)
        batch.append(img)
        if i % batch_size == 0:
            data = np.asarray(batch).reshape(batch_size, image_size)
            if y:
                targets = y[i-batch_size:i]
                yield data, targets
            else:
                yield data
            batch = []


@app.route('/metric', methods=['GET'])
def display_metrics(y_test, y_predicted):
    # Predict on validation set
    target_names = ['Dog', 'Cat']
    outcome = pd.DataFrame(confusion_matrix(y_test, y_predicted),index=target_names,
                           columns=target_names)

    print("CONFUSION MATRIX")
    print(outcome)

    report = classification_report(y_test, y_predicted, target_names=target_names)
    print("CLASSIFICATION REPORT")
    print(report)
