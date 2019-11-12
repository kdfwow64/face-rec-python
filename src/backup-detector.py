from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, prewhiten
import torch
#import tensorflow as tf
import numpy as np
import pandas as pd
import mmcv, cv2
from PIL import Image, ImageDraw, ImageFont
#from IPython import display

from fastdetector import TensoflowFaceDector, PATH_TO_CKPT

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print('Running on device: {}'.format(device))

#mtcnn = MTCNN(keep_all=True, device=device, min_face_size=100, thresholds=[0.8, 0.8, 0.8], image_size=160)
fast_det = TensoflowFaceDector(PATH_TO_CKPT)

resnet = InceptionResnetV1(pretrained='vggface2').eval()




#video = mmcv.VideoReader('1.mp4')
#frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def abs_box(box, width, height):
    return([int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height) ])


def scale_bbox(bbox, factor = 2):
    ini_w = bbox["right"] - bbox["left"]
    ini_h = bbox["bottom"] - bbox["top"]
    new_w = ini_w * factor
    new_h = ini_h * factor
    pos_x = bbox["left"] + ini_w / 2
    pos_y = bbox["top"] + ini_h / 2

    new_left = max(pos_x - new_w / 2, 0)
    new_top = max(pos_y - new_h / 2, 0)

    new_bbox = {
        "left": new_left,
        "top": new_top,
        "right": new_left + new_w, # no max width check
        "bottom": new_top + new_h, # no max height check
        "label": bbox["label"]
    }
    
    return new_bbox

old_box = []
old_emb = []

identity_dict = {}

def avg_dist(emb, embs):
    distances = [ (emb - x).norm().item() for x in embs ]
    print(np.min(distances))
    return(np.min(distances))

def propose_pid_emb(emb, id_dict):
    prop_pid = "Detecting..."
    min_avg_dist = 10000000000.0
    dict_items = [ x for x in id_dict.items() if not x[1]["detecting"] ]
    for (pid, vals) in dict_items:
        dist = avg_dist(emb, vals["embeddings"])
        if dist < 0.8 and dist < min_avg_dist:
            prop_pid = pid
            min_avg_dist = dist
    return(prop_pid)

def propose_pid_iou(box, current_frame_id, id_dict):
    prop_pid = "Detecting..."
    max_iou = 0.0
    for (pid, vals) in id_dict.items():
        if current_frame_id - vals["identity_frame"] < 8:
            iou = bb_iou(box, vals["last_box"])
            if iou > max_iou and iou > 0.4:
                prop_pid = pid
            max_iou = max(iou, max_iou)
    print(max_iou)
    return(prop_pid)           


def make_entry(box, emb, frame_id, detecting):
    vals = { "last_box" : box, "embeddings" : [emb], "identity_frame" : frame_id, "detecting": detecting }
    return(vals)

def get_emb(emb_state, image, box):
    if emb_state is not None:
        return(emb_state)
    else:
        cropped_face = extract_face(image, box)
        cropped_face = prewhiten(cropped_face)
        emb = resnet(cropped_face.unsqueeze(0))[0].detach() #.numpy().reshape(1, 512) 
        return(emb)


person_id = 0
frame_id = 0
detecting_id = 0

DETECTING = "Detecting..."

def detect_faces(frame): 
    global old_box
    global old_emb
    global person_id
    global frame_id
    global detecting_id
    emb_state = None
    frame_id += 1
    #boxes, prob = mtcnn.detect(frame)
    frame = cv2.flip(frame, 1)
    (boxes, scores, _, _) = fast_det.run(frame)
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    boxes = [ b for (s, b) in zip(scores, boxes) if s > 0.7]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    image = Image.fromarray(frame)
    
    frame_draw = image.copy()
    draw = ImageDraw.Draw(frame_draw)
    #print(prob)

    # clean detection attempts

    # for (k, v) in identity_dict.items():
    #     if v["detecting"] and frame_id - v["identity_frame"] > 20:
    #         del identity_dict[k]
    
    
    if boxes is not None:        
        #print(len(boxes))
        for box in boxes:           
            
            box = abs_box(box, frame.shape[1], frame.shape[0])
            print(box)
                       
            current_person = DETECTING
            # DB is empty - initiate first entry
            if len(identity_dict) == 0:
                current_person = "Detecting_" + str(detecting_id)
                detecting_id += 1
                emb_state = get_emb(emb_state, image, box)
                identity_dict[current_person] = make_entry(box, emb_state, frame_id, True)
            else:
                current_person = propose_pid_iou(box, frame_id, identity_dict)
                print(current_person)

                if current_person == DETECTING:
                    current_person = "Detecting_" + str(detecting_id)
                    detecting_id += 1
                    emb_state = get_emb(emb_state, image, box)
                    identity_dict[current_person] = make_entry(box, emb_state, frame_id, True)
                 
                if identity_dict[current_person]["detecting"] and len(identity_dict[current_person]["embeddings"]) > 10:
                    emb_state = get_emb(emb_state, image, box)
                    identity_dict[current_person]["embeddings"].append(emb_state)
                    identity_dict[current_person]["last_box"] = box
                    votes = [ propose_pid_emb(e, identity_dict) for e in  identity_dict[current_person]["embeddings"]]
                    (winner, nvotes) = Counter(votes).most_common(1)[0]
                    if winner != DETECTING:
                        identity_dict[winner]["embeddings"] = identity_dict[winner]["embeddings"] + identity_dict[current_person]["embeddings"]
                        identity_dict[winner]["last_box"] = box
                        del identity_dict[current_person] 
                        current_person = winner  
                    else:
                        new_person = "Person_" + str(person_id)
                        person_id += 1
                        identity_dict[new_person] = identity_dict[current_person]
                        identity_dict[new_person]["detecting"] = False
                        del identity_dict[current_person]
                        current_person = new_person
                    
                    print((winner, nvotes))
                elif identity_dict[current_person]["detecting"]:
                    emb_state = get_emb(emb_state, image, box)
                    identity_dict[current_person]["embeddings"].append(emb_state)
                    identity_dict[current_person]["last_box"] = box

                identity_dict[current_person]["last_box"] = box   
                identity_dict[current_person]["identity_frame"] = frame_id   
                if len(identity_dict[current_person]["embeddings"]) < 40:
                    emb_state = get_emb(emb_state, image, box)
                    identity_dict[current_person]["embeddings"].append(emb_state)

                
            
            #resized = scale_bbox(box)
            emb_state = None
            draw.rectangle(box, outline=(255, 0, 0), width=6)  
            fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMonoBold.ttf', 27)
            draw.text((box[0], box[3]), current_person, font=fnt, fill=(255,255,0), )
                       
    return(frame_draw)