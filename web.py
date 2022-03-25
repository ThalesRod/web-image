import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2 as cv

import imageio
from skimage import data
from skimage.color import rgb2gray
from skimage.measure import label, find_contours
from skimage.transform import rescale, resize
from skimage.morphology import erosion, dilation, square, disk

import higra as hg

import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

# draw contours on original image
def draw_contours(img, contours_to_draw):
  image_modified = np.copy(img)

  image_modified[contours_to_draw[:, 0], contours_to_draw[:, 1]] = 1

  return image_modified

def filterByCriterion(criterion = "compac", thresh = 0.55, normalize=False):

  # transform max-tree (t1) into a graph
  sources, targets = t1.edge_list()
  g2 = hg.UndirectedGraph(t1.num_vertices() - t1.num_leaves())
  g2.add_edges(sources[t1.num_leaves():] - t1.num_leaves(), targets[t1.num_leaves():] - t1.num_leaves())

  # Compute the min tree
  if criterion == "circ":
    t2 , a2 = hg.component_tree_min_tree(g2, circ)
  elif criterion == "compac":
    t2 , a2 = hg.component_tree_min_tree(g2, compac)
  elif criterion == "inertia":
    t2 , a2 = hg.component_tree_min_tree(g2, inertia)
  else:
    t2 , a2 = hg.component_tree_min_tree(g2, circ)

  # Compute the depth of the basin of g2
  depth = hg.attribute_height(t2, a2)

  # Leaves of t2 which belongs to a node with unsuficient depth
  # cond_removal = depth[t2.parents()[np.arange(t2.num_leaves())]] > thresh
  depth_nodes = depth[t2.parents()[np.arange(t2.num_leaves())]]
  depth_nodes_normalized = depth_nodes / depth_nodes.max()
  cond_removal = (depth_nodes if not normalize else depth_nodes_normalized) > thresh
  
  # transfer those t2-leaves to t1 (leaves of t2 are nodes of t1)
  cond_removal = np.concatenate((np.zeros(t1.num_leaves(), dtype=np.bool), cond_removal))

  # Reconstruct t1 
  filtered_a1 = hg.reconstruct_leaf_data(t1, a1, cond_removal)

  # Getting the number of connected components
  _, num_components = label(filtered_a1, return_num=True)

  filtered_a1_float = filtered_a1.astype(np.float32)

  # Find contours at a constant value of 0.8
  contours = find_contours(filtered_a1_float, 0.8)

  # Create an empty image to store the masked array
  filtered_a1_mask = np.zeros_like(filtered_a1_float, dtype='bool')

  for contour in contours:
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    filtered_a1_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

  # dilating contours
  dilated = dilation(filtered_a1_mask.astype(np.uint8), disk(radius=1))
  dilated_contours = np.asarray(np.where(dilated == 1)).T

  image_modified = draw_contours(image, dilated_contours)

  return image_modified, num_components


st.header("Segmetating cells")

uploaded_file = st.sidebar.file_uploader("Choose an image...")
criterion_threshold_slider = st.sidebar.slider(
                            "Criterion threshold",
                            0.0,
                            1.0,
                            0.55,
                            0.05
                            )
#col1, col2 = st.columns(2)
#
#with col1:
#    image_location = st.empty()

image_location = st.empty()

def show_image(uploaded_file):
    image_location.image(uploaded_file, caption='Image', use_column_width=True)

def load_image_gradient(uploaded_file):
    image = imageio.imread(uploaded_file)
    image = image.astype(np.float32) / 255
    image = rescale(image, 0.75, multichannel=True)

    size = image.shape[:2]

    detector = cv.ximgproc.createStructuredEdgeDetection(get_sed_model_file())
    gradient_image = detector.detectEdges(image)

    return gradient_image, size
    
    
if uploaded_file is not None:

    show_image(uploaded_file)

    gradient_image, size = load_image_gradient(uploaded_file)
    
    graph = hg.get_4_adjacency_graph(size)
    edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)

    tree, altitudes = hg.watershed_hierarchy_by_dynamics(graph, edge_weights)
    altitudes /= altitudes.max()

    explorer = hg.HorizontalCutExplorer(tree, altitudes)

    mean_color =  hg.attribute_mean_vertex_weights(tree, image)

    num_regions = 50

    cut_nodes = explorer.horizontal_cut_from_num_regions(num_regions, at_least=True)

    cut_image = cut_nodes.reconstruct_leaf_data(tree, mean_color)

    # creating binary image from cut image
    img_components = label((rgb2gray(cut_image)*255).astype(np.uint8))
    binary_label_image = img_components

    binary_label_image[binary_label_image == 1] = 0
    binary_label_image[binary_label_image > 1] = 1 # binary image

    # build the max tree
    graph = hg.get_4_adjacency_graph(size)
    t1, a1 = hg.component_tree_max_tree(graph, binary_label_image)

    # show equivalent filtered image
    filtered_a1 = hg.reconstruct_leaf_data(t1, a1)

    compac = hg.attribute_compactness(t1, normalize=True)[t1.num_leaves():]

    image_modified, num_components = filterByCriterion("compac", criterion_threshold_slider)

    st.sidebar.write("Number of resulting components:", num_components)

    image_location.image(image_modified, caption='Image', use_column_width=True)
    #st.write(os.listdir())
    # im = imgGen2(uploaded_file)	
    # st.image(im, caption='ASCII art', use_column_width=True) 
