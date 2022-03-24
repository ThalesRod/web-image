import streamlit as st
# from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt

import cv2.cv as cv

import imageio
from skimage import data
from skimage.color import rgb2gray
from skimage.measure import label, find_contours
from skimage.transform import rescale, resize
from skimage.morphology import erosion, dilation, square, disk

import higra as hg

import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())


st.header("Generate ASCII images using GAN")
st.write("Choose any image and get corresponding ASCII art:")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    # image = Image.open(uploaded_file)	

    image = imageio.imread(uploaded_file)
    image = image.astype(np.float32) / 255
    image = rescale(image, 0.75, multichannel=True)

    size = image.shape[:2]

    detector = cv.ximgproc.createStructuredEdgeDetection(get_sed_model_file())
    gradient_image = detector.detectEdges(image)

    graph = hg.get_4_adjacency_graph(size)
    edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)

    tree, altitudes = hg.watershed_hierarchy_by_dynamics(graph, edge_weights)
    altitudes /= altitudes.max()

    explorer = hg.HorizontalCutExplorer(tree, altitudes)

    mean_color =  hg.attribute_mean_vertex_weights(tree, image)

    num_regions = 50

    cut_nodes = explorer.horizontal_cut_from_num_regions(num_regions, at_least=True)
    print("Cut with %d regions at altitude %3.2f"%(len(cut_nodes.nodes()), cut_nodes.altitude()))

    cut_image = cut_nodes.reconstruct_leaf_data(tree, mean_color)

    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    st.image(cut_image, caption='Image', use_column_width=True)
    #st.write(os.listdir())
    # im = imgGen2(uploaded_file)	
    # st.image(im, caption='ASCII art', use_column_width=True) 
