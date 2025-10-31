import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, img, alpha=0.4, cmap='viridis'):
    plt.imshow(img, cmap='gray')
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("gradcam_overlay.png")
    plt.close()
    return "gradcam_overlay.png"
