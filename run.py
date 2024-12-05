import tensorflow as tf
import tflite_runtime.interpreter
import cv2
from utils import cxy_wh_2_rect, hann1d, hann2d, img2tensor
import numpy as np
import math
import time
import torch

class Tracker(object):
    def __init__(self,model_path:str) -> None:
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.template_factor = 2.0
        self.search_factor = 4.0
        self.template_size = 128
        self.search_size = 256
        self.stride = 16
        self.feat_sz = self.search_size // self.stride
        self.output_window = hann2d(np.array([self.feat_sz, self.feat_sz]), centered=True)
        self.z = None
        self.state = None

    def initialize(self, image, target_bb):
        # get subwindow
        z_patch_arr, resize_factor, z_amask_arr = self.sample_target(image, target_bb, self.template_factor,
                                                    output_sz=self.template_size)
        # nparry -> onnx input tensor
        self.z = img2tensor(z_patch_arr)
        # get box_mask_z
        self.box_mask_z = self.generate_mask_cond()
        # save states
        self.state = target_bb


    def track(self, image):
        img_H, img_W, _ = image.shape

        # Get subwindow
        x_patch_arr, resize_factor, x_amask_arr = self.sample_target(image, self.state, self.search_factor,
                                                                    output_sz=self.search_size)
                                                                    
        # Prepare input tensor with expected shape
        x = img2tensor(x_patch_arr)
        # Ensure that `self.z` and `x` match the required input shapes
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # print(input_details)
        # print(output_details)
        # print(self.z.shape)
        # print(x.shape)

        z_input = np.transpose(self.z, (0, 2, 3, 1)).astype(np.float32)  # Shape will become [1, 128, 128, 3]
        x_input = np.transpose(x, (0, 2, 3, 1)).astype(np.float32)       # Shape will become [1, 256, 256, 3]
        # print(z_input.shape)
        # print(x_input.shape)

        # Set the reshaped tensors
        self.interpreter.set_tensor(input_details[0]['index'], z_input)
        self.interpreter.set_tensor(input_details[1]['index'], x_input)

        # Run inference

        self.interpreter.invoke()
        
        # Get output maps
        out_score_map = self.interpreter.get_tensor(output_details[1]['index'])
        out_size_map = self.interpreter.get_tensor(output_details[2]['index'])
        out_offset_map = self.interpreter.get_tensor(output_details[3]['index'])

        print("Score map shape:", out_score_map)
        quit()
        # print("Size map shape:" ,out_size_map.shape)
        # print("Offset map shape:", out_offset_map.shape)

        out_score_map = np.transpose(out_score_map,(0,3,1,2)) 
        out_offset_map = np.transpose(out_offset_map,(0,3,1,2)) 
        out_size_map = np.transpose(out_size_map,(0,3,1,2))

        
        # Apply Hann window
        response = self.output_window * out_score_map

        # Calculate predicted bounding box
        pred_boxes = self.cal_bbox(response, out_size_map, out_offset_map)
        pred_box = (pred_boxes * self.search_size / resize_factor).tolist()
        
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), img_H, img_W, margin=10)
        
        return self.state


    
    def sample_target(self, im, target_bb, search_area_factor, output_sz):
        """Extracts a square crop centered at target_bb box, of are search_area_factor^2 times target_bb area
        args: 
            im - cv image
            target_bb - target box [x_left, y_left, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = list(target_bb)
        else:
            x, y , w, h = target_bb
        # crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception("Too small bounding box.")
        
        cx, cy = x + 0.5 * w, y + 0.5 * h
        x1 = round(cx - crop_sz * 0.5)
        y1 = round(cy - crop_sz * 0.5)

        x2 = x1 + crop_sz
        y2 = y1 + crop_sz 

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        
        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H,W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0

        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz))

        return im_crop_padded, resize_factor, att_mask
    
    def transform_bbox_to_crop(self, box_in: list, resize_factor, crop_type='template', normalize=True) -> list:
        """Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
        args:
            box_in: list [x1, y1, w, h], not normalized, the box for which the co-ordinates are to be transformed
            resize_factor - the ratio between the original image scale and the scale of the image crop

        returns:
            List - transformed co-ordinates of box_in
        """
        
        if crop_type == 'template':
            crop_sz = self.template_size
        elif crop_type == 'search':
            crop_sz = self.search_size
        else:
            raise NotImplementedError
        
        box_out_center_x = (crop_sz[0] - 1) / 2
        box_out_center_y = (crop_sz[1] - 1) / 2
        box_out_w = box_in[2] * resize_factor
        box_out_h = box_in[3] * resize_factor

        # normalized
        box_out_x1 = (box_out_center_x - 0.5 * box_out_w)
        box_out_y1 = (box_out_center_y - 0.5 * box_out_h)
        box_out = [box_out_x1, box_out_y1, box_out_w, box_out_h]

        if normalize:
            return [i / crop_sz for i in box_out]
        else:
            return box_out
        
    def generate_mask_cond(self):
        template_size = self.template_size
        stride = self.stride
        template_feat_size = template_size// stride # 128 // 16 = 8

        # MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT'

        box_mask_z = np.zeros([1, template_feat_size, template_feat_size])
        box_mask_z[:, slice(3, 4), slice(3, 4)] = 1
        box_mask_z = np.reshape(box_mask_z, (1, -1)).astype(np.int32)

        return box_mask_z
    

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        # Convert TFLite outputs to numpy arrays
        # score_map_ctr = tf.convert_to_tensor(score_map_ctr, dtype=tf.float32)
        # size_map = tf.convert_to_tensor(size_map, dtype=tf.float32)
        # offset_map = tf.convert_to_tensor(offset_map, dtype=tf.float32)
        
        # Find the maximum score index
        # max_score = tf.reduce_max(score_map_ctr, axis=[1, 2, 3])
        # idx = tf.argmax(tf.reshape(score_map_ctr, [score_map_ctr.shape[0], -1]), axis=1)
    
        flattened_score_map_ctr = tf.reshape(score_map_ctr, [score_map_ctr.shape[0], -1])

        # Get the max value and index across axis 1, keeping the dimension
        max_score, idx = tf.reduce_max(flattened_score_map_ctr, axis=1, keepdims=True), tf.argmax(flattened_score_map_ctr, axis=1, output_type=tf.int32)
        
        # Compute (x, y) coordinates
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        
         # Reshape idx to match ONNX output format (shape [1, 2, 1])
        idx = tf.reshape(idx, (-1, 1))  # Reshape to [batch_size, 1]
        idx = tf.tile(idx, [1, 2])  # Expand idx to [batch_size, 2]
        idx = tf.expand_dims(idx, axis=-1)  # Unsqueeze to get shape [batch_size, 2, 1]
        size_map_flat = tf.reshape(size_map, [size_map.shape[0], -1, 2])
        offset_map_flat = tf.reshape(offset_map, [offset_map.shape[0], -1, 2])

        size_map_flat=tf.transpose(size_map_flat,perm=[0,2,1])
        offset_map_flat = tf.transpose(offset_map_flat,perm=[0,2,1])

        size = tf.gather(size_map_flat, idx, batch_dims=2)
        offset = tf.gather(offset_map_flat, idx, batch_dims=2)
        offset= tf.squeeze(offset, axis=-1)
        # Calculate the bounding box
        bbox = tf.concat([
            (tf.cast(idx_x, tf.float32) + offset[:, :1]) / self.feat_sz,
            (tf.cast(idx_y, tf.float32) + offset[:, 1:]) / self.feat_sz,
            tf.squeeze(size,axis=-1)
        ], axis=1)

        bbox = bbox.numpy()[0]
        if return_score:
            return bbox, max_score.numpy()
        return bbox

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
    
    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W-margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H-margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2-x1)
        h = max(margin, y2-y1)
        return [x1, y1, w, h]
    
def run(tracker, video_path):
    '''
    tracker: mobilevit-track
    video_path: 0 for webcam or a video file path
    '''
    cap = cv2.VideoCapture(video_path)

    # if not cap.isOpened():
    #     print("Error: Could not open video.")
    #     return

    # frame_count = 0
    # bbox = None

    
    # Get the width and height of frames from the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use 'MJPG', 'X264', etc.
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    bbox = None
    prev_time = time.time()  # To calculate FPS
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_track = frame.copy()

        # Initialize the tracker with the first frame and bounding box
        if frame_count == 0:
            # Allow the user to select a bounding box on the first frame
            # bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            # Ensure the tracker gets initialized with the selected bounding box
            # print(bbox)
            bbox = 454, 21, 349, 438
            tracker.initialize(frame_track, bbox)
        else:
            # Track the object in the subsequent frames
            bbox = tracker.track(frame_track)
        
        x, y, w, h = bbox
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        location = cxy_wh_2_rect(target_pos, target_sz)

        # Draw the bounding box
        x1, y1, x2, y2 = int(location[0]), int(location[1]), \
            int(location[0] + location[2]), int(location[1] + location[3])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        current_time = time.time()
        fps_tracker = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps_tracker:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the tracking result
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mvt_track = Tracker(model_path="MobileViT_Track_ep0300_float16.tflite")
    video_path = '/home/soumy/test.mp4'  # Replace with your video file path or 0 for webcam
    run(mvt_track, video_path)
