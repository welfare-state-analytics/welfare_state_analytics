# # %%
# from  transformers import pipeline
# deps:
# transformers = "^4.3.2"
# torch = "^1.7.1"
# tensorflow = "^2.4.1"
# # https://github.com/search?l=Python&q=bert-base-swedish-cased-ner&type=Code

# def create_pipeline():
#     model = "KB/bert-base-swedish-cased-ner"
#     tokenizer = "KB/bert-base-swedish-cased-ner"
#     nlp = pipeline("ner", model=model, tokenizer=tokenizer)

#     entities = [nlp(sentence) for sentence in sentences]

# # %%

# import tensorflow as tf
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# # Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
# #tf.config.experimental.set_memory_growth(physical_devices[0], True)
# #[![enter image description here][1]][1]

# # %%
