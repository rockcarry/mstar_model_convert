import math

import numpy as np
import six
import pdb

def genAnchors(layer_width,
        layer_height,
        img_width,
        img_height,
        aspect_ratios,
        variance,
        offset,
        num_prior,
        min_sizes,
        max_sizes,
        clip,
        step_h=0,
        step_w=0,
        img_h=0,
        img_w=0):
    """Generate SSD Prior Boxes.
        It returns the center, height and width of the priors. The values are relative to the image size
        Returns:
            priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                are relative to the image size.
    """
    img_width_,img_height_ = 0,0
    if (img_h == 0 or img_w == 0):
      img_width_ = img_width
      img_height_ = img_height
    else:
      img_width_ = img_w
      img_height_ = img_h
    top_data = np.array([])
    scale_x = img_width_ / step_w
    scale_y = img_height_ / step_h
    #pdb.set_trace()
    for i in six.moves.range(layer_height):
      for j in six.moves.range(layer_width):
        center_x = (j + offset) / scale_x
        center_y = (i + offset) / scale_y
        box_width,box_height = 0,0
        for s in six.moves.range(len(min_sizes)):
          min_size_ = min_sizes[s]
          # small sized square box
          #box_width = box_height = min_size_
          #w
          #w = box_width / img_width_
          #h
          #h = box_height / img_height_
          #top_data = np.append(top_data,[center_x,center_y,w,h])
          # big sized square box
          if len(max_sizes)>0:
            max_size_ = max_sizes[s]
            box_width = box_height = math.sqrt(min_size_ * max_size_)
            #w
            w = box_width / img_width_
            #h
            h = box_height / img_height_
            top_data = np.append(top_data,[center_x,center_y,w,h])

          # change h/w ratio of the small sized box
          for r in six.moves.range(len(aspect_ratios)):
            ar = aspect_ratios[r]
            ar = math.sqrt(ar)
            box_width = box_height = min_size_
            #w
            w = box_width / img_width_
            #h
            h = box_height / img_height_
            top_data = np.append(top_data,[center_x,center_y,w * ar, h / ar])
            top_data = np.append(top_data,[center_x,center_y,w / ar, h * ar])
    '''
    dim = layer_width * layer_height * num_prior * 4
    if(clip):
      for i in six.movies.range(dim):
        top_data = math.min(math.max(top_data[i],0.),1.)

   #set the variance
    for h in six.moves.range(layer_height):
      for w in six.moves.range(layer_width):
        for i in six.moves.range(num_prior):
          for j in six.moves.range(4):
            top_data = np.append(top_data,variance[j])
    '''
    if(clip):
        top_data = np.clip(top_data , 0, 1)
    #top_data.reshape(-1,1)
    #np.savetxt('./priors.txt',top_data)
    #pdb.set_trace()
    return top_data
