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
    #set img w and h
    #pdb.set_trace()
    img_width_,img_height_ = 0,0
    if (img_h == 0 or img_w == 0):
      img_width_ = img_width
      img_height_ = img_height
    else:
      img_width_ = img_w
      img_height_ = img_h
    #set step
    step_h_,step_w_ = 0,0
    if (step_h == 0 or step_w == 0):
      step_h_ = img_width / layer_width
      step_w_ = img_height / layer_height
    else:
      step_h_ = step_h
      step_w_ = step_w
    top_data = np.array([])
    dim = layer_width * layer_height * num_prior * 4
    for h in six.moves.range(layer_height):
      for w in six.moves.range(layer_width):
        center_x = (w + offset) * step_w_
        center_y = (h + offset) * step_h_
        box_width,box_height = 0,0
        for s in six.moves.range(len(min_sizes)):
          min_size_ = min_sizes[s]
          #first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size_;
          #xmin
          xmin = (center_x - box_width / 2.) / img_width_
          #ymin
          ymin = (center_y - box_height / 2.) / img_height_
          #xmax
          xmax = (center_x + box_width / 2.) / img_width_
          #ymax
          ymax = (center_y + box_height / 2.) / img_height_
          top_data = np.append(top_data,[xmin,ymin,xmax,ymax])
          if len(max_sizes)>0:
            max_size_ = max_sizes[s]
            #ptype == CLASSICAL_PRIOR
            box_width = box_height = math.sqrt(min_size_ * max_size_)
            #xmin
            xmin = (center_x - box_width / 2.) / img_width_
            #ymin
            ymin = (center_y - box_height / 2.) / img_height_
            #xmax
            xmax = (center_x + box_width / 2.) / img_width_
            #ymax
            ymax = (center_y + box_height / 2.) / img_height_
            top_data = np.append(top_data,[xmin,ymin,xmax,ymax])

          #rest of priors
          for r in six.moves.range(len(aspect_ratios)):
            ar = aspect_ratios[r]
            if math.fabs(ar - 1.) < 1e-6:
              continue
            box_width = min_size_ * math.sqrt(ar)
            box_height = min_size_ / math.sqrt(ar);
            #xmin
            xmin = (center_x - box_width / 2.) / img_width_
            #ymin
            ymin = (center_y - box_height / 2.) / img_height_
            #xmax
            xmax = (center_x + box_width / 2.) / img_width_
            #ymax
            ymax = (center_y + box_height / 2.) / img_height_
            top_data = np.append(top_data,[xmin,ymin,xmax,ymax])
    '''
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
    return top_data
'''
layer_width = layer_height = 19
img_width = img_height = 300
aspect_ratios = [1.0,2.0,0.5]
variance = [0.1,0.1,0.2,0.2]
offset = 0.5
min_sizes = [60.0]
max_sizes = []
num_prior = len(aspect_ratios) * len(min_sizes) + len(max_sizes)
top = genAnchors(layer_width,layer_height,img_width,img_height,aspect_ratios,variance,offset,num_prior,min_sizes)
pdb.set_trace()
'''