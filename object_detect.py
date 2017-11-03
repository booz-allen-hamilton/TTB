import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Display plots inline
%matplotlib inline
from selectivesearch import selective_search
import PIL
from PIL import Image
import pandas as pd
import os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def normalize_region(region, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  obj_tensor = tf.stack(region)
  type(obj_tensor)
  float_caster = tf.cast(obj_tensor, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  type(dims_expander)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  type(normalized)
  sess = tf.Session()
  result = sess.run(normalized)
  type(result)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def predict(graph_model,labels, norm_reg):
    with tf.Session(graph=graph_model) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: norm_reg})
    results = np.squeeze(results)
    top = results.argsort()[::-1][0]
    predicted_label = labels[top]
    conf_score = results[top]
    return [predicted_label, conf_score]

# Perform non-maximum suppression to greedily merge detections with highest confidence scores
def nms(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detections
    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'class_index']
    overlap: float
        minimum overlap ratio (0.3 default)
    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

# Create environment folders
OD_DIR = "./Outputs/OD_results"
RP_DIR = "./Outputs/RP_Candidates"
if not os.path.exists(OD_DIR):
    os.makedirs(OD_DIR)
if not os.path.exists(RP_DIR):
    os.makedirs(RP_DIR)

# Initialize Classifier Parameters
model_file = "./retrained_graph_inception.pb"
label_file = "./retrained_labels_inception.txt"
input_height = 299
input_width = 299
input_mean = 128
input_std = 128
input_layer = "Mul"
output_layer = "final_result"
conf_thresh = 0.6

# Load Classifier
graph = load_graph(model_file)
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);

# Load labels
labels = load_labels(label_file)

# Get test image
TEST_IMAGE = './Bottle_labels/Bottle11.jpg'
IMAGE_NAME = (TEST_IMAGE.split("/")[-1]).split(".")[0]

# Resize the image
baseheight = 560
img = Image.open(TEST_IMAGE)
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
input_image = np.array(img)

# Use Selective Search to find all potential objects
img_lbl, regions = selective_search(input_image, scale=500, sigma=0.9, min_size=10)

# Prune the list of object candidates
candidates = set()
for r in regions:
        # Exclude same rectangle (with different segments)
        if r['rect'] in candidates: continue
        # Exclude small or large regions
        if r['size'] < 2000 or r['size'] > 50000: continue
        # Exclude wide or long distorted rectangles
        x, y, w, h = r['rect']
        if w / h > 2.0 or h / w > 2.0: continue
        candidates.add(r['rect'])

print('Number of object candidates: {:d}'.format(len(candidates)))

# Draw object candidates on the original image
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.imshow(input_image)
for x, y, w, h in candidates:
    rect = patches.Rectangle((x, y), w, h, fill=False,edgecolor='red', linewidth=1)
    ax.add_patch(rect)
# Save image to "Outputs/RP_Candidates" folder
plt.savefig('./Outputs/RP_Candidates/'+IMAGE_NAME+'.jpg')

# Perform NMS on predicted objects
dets = []
for obj in candidates:
    x, y, w, h = obj
    obj = input_image[y:y+h, x:x+w]
    norm_reg = normalize_region(obj)
    # Make prediction on each object candidate
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: norm_reg})
    results = np.squeeze(results)
    top_index = results.argsort()[::-1][0]
    predicted_label = labels[top_index]
    conf_score = results[top_index]
    # Label ID 3 is "Others"
    if top_index == 3 or conf_score < conf_thresh:
        continue
    dets.append(np.hstack([x, y, x+w, y+h, conf_score, top_index]))

try:
    dets = nms(np.vstack(dets), overlap=0.1)
    # Visualize nms results
    print('NMS Detections')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(input_image)

    for det in dets:
        xmin, ymin, xmax, ymax = map(int, det[:4])
        conf_score = float(det[4])
        predicted_index = int(det[5])
        rect = patches.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin,
                                         linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, labels[predicted_index] + ': {:.3f}'.format(conf_score), style='normal', color='white',
                        bbox={'facecolor':'green', 'edgecolor':'green', 'alpha':0.8, 'pad':2})
    # Save image to "Outputs/OD_Results" folder
    plt.savefig('./Outputs/OD_results/'+IMAGE_NAME+'.jpg')
except:
    print("No violation found")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(input_image)
    plt.savefig('./Outputs/OD_results/'+IMAGE_NAME+'.jpg')

df = pd.DataFrame(dets, columns=['Xmin','Ymin','Xmax','Ymax','conf_score','class'])
# Map class index class label
mapping = {0: 'usflag',1: 'gmo',2: 'children',3: 'others'}
df['class'] = df['class'].map(mapping)
# Find max value
reduced_df = df.groupby(['class'],sort=False)['conf_score'].max()
reduced_df.to_csv('./Outputs/OD_results/'+IMAGE_NAME+'.csv')
print(reduced_df)
#reduced_df.to_csv().replace("\n",",")
