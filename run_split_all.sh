#!/bin/bash
python split_embedings.py "Embeddings Generated Arcface/CASIA-WebFace-400K/arcface_embeddings.npy"
python split_embedings.py "Embeddings Generated AdaFace/CASIA-WebFace-400K/adaface_embeddings.npy"
python split_embedings.py "Embeddings Generated CosFace/CASIA-WebFace-400K/cosface_embeddings.npy"
python split_embedings.py "Embeddings Generated DeepFace/CASIA-WebFace-400K/deepface_embeddings.npy"
python split_embedings.py "Embeddings Generated FaceNet/CASIA-WebFace-400K/facenet_embeddings.npy"
python split_embedings.py "Embeddings Generated FaceT/CASIA-WebFace-400K/facet_embeddings.npy"
python split_embedings.py "Embeddings Generated MagFace/CASIA-WebFace-400K/magface_embeddings.npy"
python split_embedings.py "Embeddings Generated SphereFace/CASIA-WebFace-400K/sphereface_embeddings.npy"
python split_embedings.py "Embeddings Generated TransFace/CASIA-WebFace-400K/transface_embeddings.npy"
python split_embedings.py "Embeddings Generated ViT/CASIA-WebFace-400K/vit_embeddings.npy"

