import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt

core = ov.Core()

model_face = core.read_model(model='models/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model_face, device_name="CPU")

input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)


model_person = core.read_model(model='models/person-detection-retail-0013.xml')
compiled_model_person = core.compile_model(model_person, device_name="CPU")

input_layer_person = compiled_model_person.input(0)
output_layer_person = compiled_model_person.output(0)


def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    
    return input_image

def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()
    
    scores = results[:, 2]
    boxes = results[:, -4:]
    
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    
    image_h, image_w, _ = image.shape
    
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)
    
    return face_boxes, scores

def find_personboxes(image, results, confidence_threshold):
    results = results.squeeze()
    
    scores = results[:, 2]
    boxes = results[:, -4:]
    
    person_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    
    image_h, image_w, _ = image.shape
    person_boxes = person_boxes * np.array([image_w, image_h, image_w, image_h])
    person_boxes = person_boxes.astype(np.int64)
    
    return person_boxes, scores

def draw_face_and_person_boxes(face_boxes, person_boxes, frame):
    show_frame = frame.copy()
    fontScale = frame.shape[1] / 500  # Adjust font size based on frame width

    # Draw bounding boxes for persons
    for i in range(len(person_boxes)):
        xmin, ymin, xmax, ymax = person_boxes[i]
        cv2.rectangle(show_frame, (xmin, ymin), (xmax, ymax), color=(0, 200, 0), thickness=5)
        text = 'Person'
        cv2.putText(show_frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 2)

    # Draw bounding boxes for faces
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        cv2.rectangle(show_frame, (xmin, ymin), (xmax, ymax), color=(0, 0, 200), thickness=5)
        text = 'Face'
        cv2.putText(show_frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 200), 2)

    return show_frame

def predict_image(image, conf_threshold):
    # Preprocess the image for face detection
    input_image_face = preprocess(image, input_layer_face)
    results_face = compiled_model_face([input_image_face])[output_layer_face]

    # Preprocess the image for person detection
    input_image_person = preprocess(image, input_layer_person)
    results_person = compiled_model_person([input_image_person])[output_layer_person]

    # Find face and person boxes
    face_boxes, _ = find_faceboxes(image, results_face, conf_threshold)
    person_boxes, _ = find_personboxes(image, results_person, conf_threshold)

    # Draw the bounding boxes on the image
    visualize_image = draw_face_and_person_boxes(face_boxes, person_boxes, image)

    return visualize_image
