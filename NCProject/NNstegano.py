import math
from pathlib import Path
from copy import deepcopy
import struct
import bitstring
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time


start = time.time()
#We are modifying only the last 16 bits because we are using least significant bit technique
BITS_TO_USE = 16
# we are loading our model
model = tf.keras.applications.ResNet152(include_top=True, weights="imagenet")
#this dictionary we are using to store the capacity of data that can be stored in each layer
capacity= {}
j=0
for l in model.layers:
    if l.__class__.__name__ == "Conv2D":
        nb_params = np.prod(l.get_weights()[0].shape)
        capacity_in_bytes = np.floor((nb_params * BITS_TO_USE) / 8).astype(int)
        capacity[l.name] = capacity_in_bytes / float(1<<20)
        j=j+1
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.bar(capacity.keys(), capacity.values())
ax.tick_params(axis='x', labelrotation = 90)
ax.set_xlabel("Layer Number")
ax.set_ylabel("Megabytes")
ax.set_title(f"Storage capacity when using {BITS_TO_USE} bits from every float value")
print(plt.show())
#storing the name  of layers
layer_names = list(capacity.keys())
#here we are storing the weights of all convoluted 2d layers and printing the statistics related to it
selected_layers_weights = []
for n in layer_names:
    v = model.get_layer(n).weights[0].numpy().ravel()
    selected_layers_weights.extend(v)
selected_layers_weights = np.array(selected_layers_weights)
nb_values = len(selected_layers_weights)
min_value = selected_layers_weights.min()
abs_min_value = np.abs(selected_layers_weights).min()
max_value = selected_layers_weights.max()
mean_value = selected_layers_weights.mean()
nb_really_small_values = (abs(selected_layers_weights) < 10e-4).sum()
nb_small_values = (abs(selected_layers_weights) < 10e-3).sum()
nb_negative_values = (selected_layers_weights < 0).sum()
nb_positive_values = (selected_layers_weights > 0).sum()
overall_storage_capacity_bytes = nb_values * BITS_TO_USE / 8
overall_storage_capacity_mb = overall_storage_capacity_bytes // float(1<<20)
print(f"""Statistics related to the weights for ResNet152 model
---
Min weight: {min_value}
Abs. Min weight {abs_min_value}
Max weight: {max_value}
Mean of all weightd : {mean_value}
---
Total number of weights: {nb_values}
weight values < 10e-4: {nb_really_small_values} - {nb_really_small_values/nb_values*100:.4f}%
weight values < 10e-3: {nb_small_values} - {nb_small_values/nb_values*100:.4f}%
Total negative weights: {nb_negative_values} - {nb_negative_values/nb_values*100:.4f}%
Total positive weights: {nb_positive_values} - {nb_positive_values/nb_values*100:.4f}%
---
(Maximum) Storage capacity is {overall_storage_capacity_mb} MB for the {len(layer_names)} layers with the {BITS_TO_USE} bits modification
""")

fi=open('cipher.txt','r')
secret_bits=""
de=['']
c=0
count=0
for i in fi:
    for j in str(i):
        if(j=='@'):
            c+=1
            de.append('')
        else:
            de[c]+=str(j)
            count=count+1
de.pop()
for i in de:
    f1 = bitstring.BitArray(float=float(i), length=32)
    secret_bits=secret_bits+str(f1.bin)
nb_vals_needed = math.ceil(len(secret_bits) / BITS_TO_USE)
print(f"We need {nb_vals_needed} float values to store the info\nOverall number of values we could use: {nb_values}")
# This dict holds the original weights for the selected layers
original_weights_dict= {}
for n in layer_names:
    original_weights_dict[n] = deepcopy(model.get_layer(n).weights[0].numpy())
modified_weights_dict = deepcopy(original_weights_dict)
last_index_used_in_layer_dict= {}
i = 0

for n in layer_names:
    # Check if we need more values to use to hide the secret, if not then we are done with modifying the layer's weights
    if i >= nb_vals_needed:
        break

    w = modified_weights_dict[n]
    w_shape = w.shape
    w = w.ravel()

    nb_params_in_layer = np.prod(w.shape)

    for j in range(nb_params_in_layer):
        # Chunk of data from the secret to hide
        _from_index = i * BITS_TO_USE
        _to_index = _from_index + BITS_TO_USE
        bits_to_hide = secret_bits[_from_index:_to_index]
        f1 = bitstring.BitArray(float=w[j], length=32)
        fraction_modified = list(f1.bin)
        if len(bits_to_hide) > 0:
            fraction_modified[-BITS_TO_USE:] = bits_to_hide
        listToStr = ''.join([str(elem) for elem in fraction_modified])
        f=int(listToStr,2)
        x=struct.unpack('f', struct.pack('I', f))[0]
        w[j] = x
        i += 1   
        # Check if we need more values to use to hide the secret in the current layer, if not then we are done
        if i >= nb_vals_needed:
            break
    last_index_used_in_layer_dict[n] = j
    w = w.reshape(w_shape)
    modified_weights_dict[n] = w
    print(f"Layer {n} is processed, last index modified: {j}")
#storing the path of the file related to data
DATA_FOLDER = "C:/Users/saikr/Downloads/NCProject/data"
IMAGES_TO_TEST_ON = list(map(str, Path(DATA_FOLDER).glob("**/*.jpg")))
assert len(IMAGES_TO_TEST_ON) > 0, "You'll need some images to test the network performance"

batch = 8
def _read_image_from_path(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, dtype=tf.uint8, expand_animations=False)
    image = tf.image.resize(image, (224, 224))
    return image
dataset = tf.data.Dataset.from_tensor_slices(IMAGES_TO_TEST_ON)
dataset = dataset.map(_read_image_from_path, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch).prefetch(tf.data.AUTOTUNE)
for n in layer_names:
    w = original_weights_dict[n]
    model.get_layer(n).set_weights([w, model.get_layer(n).get_weights()[1]])
preds_original = model.predict(dataset)
# Load the modified (secret is hidden) weights to the model layers
for n in layer_names:
    w = modified_weights_dict[n]
    model.get_layer(n).set_weights([w, model.get_layer(n).get_weights()[1]])
preds_modified = model.predict(dataset)
diff_abs = np.abs(preds_original - preds_modified).ravel()
diff_value=mean_squared_error(preds_original.ravel(),preds_modified.ravel())
print("mean squared error value", diff_value)
# plt.hist(diff_abs[diff_abs > 0])
# print(plt.show())
# print(f"Min abs difference: {diff_abs.min()}")
# print(f"Max abs difference: {diff_abs.max()}")
# print(f"Number of changed prediction values: {(diff_abs > 0).sum()} / {len(diff_abs)} | {(diff_abs > 0).sum()/len(diff_abs)*100:.4f}%")
# nb_changed_pred_labels = ((np.argmax(preds_original, 1) - np.argmax(preds_modified, 1)) > 0).sum()
# print(f"Changed number of predictions: {nb_changed_pred_labels} / {len(IMAGES_TO_TEST_ON)} | {nb_changed_pred_labels / len(IMAGES_TO_TEST_ON)*100}%")
# # We store the extracted bits of data here
hidden_data= []
for n in layer_names:
    # Check if the layer was used in hiding the secret or not (e.g.: we could hide the secret in the prev. layers)
    if n not in last_index_used_in_layer_dict.keys():
        continue
    # We could get the modified weights directly from the model: model.get_layer(n).get_weights()...
    w = modified_weights_dict[n]
    w_shape = w.shape
    w = w.ravel()
    nb_params_in_layer: int = np.prod(w.shape)
    for i in range(last_index_used_in_layer_dict[n]+1):
        f1 = bitstring.BitArray(float=w[i], length=32)
        hidden_bits = f1.bin[BITS_TO_USE:]
        listToStr = ''.join([str(elem) for elem in hidden_bits])
        hidden_data.append(hidden_bits)
ciptxt=''
for i in range(0,len(hidden_data),2):
    sk=hidden_data[i]+hidden_data[i+1]
    f=int(sk,2)
    x=struct.unpack('f', struct.pack('I', f))[0]
    ciptxt+=str(x)+'@'
fi=open('NNcipher.txt','w+')
fi.write(ciptxt)
fi.close()
end = time.time()
total_time = end-start
print(total_time)