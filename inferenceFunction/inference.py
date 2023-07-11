

mediapipe = None
yolo = None


def setup_mediapipe():
    import modelOne.mediapipe
    global mediapipe
    mediapipe = modelOne.mediapipe
    return  mediapipe

def setup_yolo():
    import modelTwo.yolo
    global yolo
    yolo = modelTwo.yolo
    return yolo

def has_hand(frame, thresh, model):
    global mediapipe
    global yolo
    if model == "mediapipe":
        if mediapipe == None:
            mediapipe = setup_mediapipe()
            if mediapipe.mediapipe_for_images(frame):
                return True
            return False
        elif mediapipe.mediapipe_for_images(frame):
            return True
        return False
    elif model == "handobj":
        if yolo == None:
            yolo = setup_yolo()
            result = yolo.yolo_for_image(frame)
            for _, obj in result.pandas().xyxy[0].iterrows():
                if obj["name"] == "hand" and obj["confidence"] > thresh:
                    return True
            return False
        else:
            result = yolo.yolo_for_image(frame)
            for _, obj in result.pandas().xyxy[0].iterrows():
                if obj["name"] == "hand" and obj["confidence"] > thresh:
                    return True
            return False


IMAGE_FILES = ['universal-events-identifies-counterproductive-habits-blog-image.jpg','photo_2023-07-11_22-09-45.jpg'
               ,'1614344187_55-p-chelovek-na-svetlom-fone-59.jpg']

for frame in IMAGE_FILES:
    print(has_hand(frame, 0.8, "mediapipe"))