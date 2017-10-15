#!/usr/bin/python3

import os
import sys

import dlib
from skimage import io
from scipy.spatial import distance
import cv2


def initialize():
    global sp, facerec, detector
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = (dlib.rectangle(
                left=int(d[0]),
                top=int(d[1]),
                right=int(d[0]+d[2]),
                bottom=int(d[1]+d[3])
            ) for d in detector.detectMultiScale(gray))
    return [(sp(img, d), d) for d in faces]


def extract_features(img, faces):
    return [facerec.compute_face_descriptor(img, face[0]) for face in faces]


def read_database(path):
    db = {}
    for name in os.listdir(path):
        img = io.imread(path + '/' + name)
        db[name[:name.index('.')]] = extract_features(img, extract_faces(img))[0]
    return db


def draw_box(img, box, color):
    cv2.rectangle(img, (box.left(), box.top()), (box.right(), box.bottom()), color, 1)


def main():
    initialize()
    db = read_database('database')

    img = io.imread(sys.argv[1])
    faces = extract_faces(img)
    features = extract_features(img, faces)

    for face, vec in zip(faces, features):
        draw_box(img, face[1], (0, 0, 255))

        best = (0.6, '')
        for name, cand in db.items():
            best = min(best, (distance.euclidean(vec, cand), name))
        if not best[1]:
            continue

        draw_box(img, face[1], (255, 0, 0))
        cv2.putText(img, best[1], (face[1].left(), face[1].top() - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)

    input('\nPress any key to continue')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ./recognizer.py [path to photo]')
        exit()
    main()
