#!/bin/bash

python query_metrices_all.py "Embeddings Generated Arcface/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated AdaFace/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated CosFace/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated DeepFace/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated FaceNet/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated FaceT/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated MagFace/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated SphereFace/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated TransFace/CASIA-WebFace-400K/Split/reference_embeddings.npy"
python query_metrices_all.py "Embeddings Generated ViT/CASIA-WebFace-400K/Split/reference_embeddings.npy"

