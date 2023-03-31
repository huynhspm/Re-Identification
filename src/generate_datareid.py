import os
import cv2
import glob
import time
import shutil
import random
from utils import get_object_frame


def create_folder(path):
    if not os.path.exists(path):
        print("Create folder: ", path)
        os.mkdir(path)


def check_num_files(path):
    if os.path.exists(path):
        return len(os.listdir(path))
    return 0


def split_train_gallery(train_path, gallery_path, ratio=1):
    """ratio(float): split ratio between train dataset and gallery dataset
        (default=1) just for test reid model"""

    track_folder = os.listdir(train_path)
    num_id_gallery = int(len(track_folder) * ratio)

    random.shuffle(track_folder)

    for t in track_folder:
        if (num_id_gallery == 0):
            break
        save_folder = os.path.join(gallery_path, t)
        target_folder = os.path.join(train_path, t)

        create_folder(save_folder)
        imgs = glob.glob(target_folder + "/*.jpg")

        if len(imgs) == 1:
            continue

        num_test = min(5, int(len(imgs) * ratio))
        num_test = max(num_test, 1)
        test_img_ls = random.sample(imgs, num_test)

        for img in test_img_ls:
            shutil.move(img, save_folder)

        num_id_gallery -= 1


def split_gallery_query(query_path, gallery_path, query_sample=1):
    track_folder = os.listdir(gallery_path)

    for t in track_folder:
        save_folder = os.path.join(query_path, t)
        target_folder = os.path.join(gallery_path, t)

        imgs = glob.glob(target_folder + "/*.jpg")

        create_folder(save_folder)
        if (len(imgs) == 1):
            continue

        query_img_ls = random.sample(imgs, query_sample)
        for img in query_img_ls:
            shutil.move(img, save_folder)


def generate_frames(video_path, train_path, annotation_path):
    vidCapture = cv2.VideoCapture(video_path)
    success, image = vidCapture.read()

    if (success):
        print("Capture video successfully")
    else:
        print("Failed")

    fps = vidCapture.get(cv2.CAP_PROP_FPS)
    print("Frame rate: ", fps)

    frames = vidCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Number of frames: ", frames)

    print(video_path)
    cam_id = video_path.split('/')[-1][:6]
    duration_id = video_path.split('/')[-2]
    print("annotation path: ", annotation_path)
    group_frame = get_object_frame(annotation_path, frame_start=0)

    while success:
        # current frame number, rounded b/c sometimes you get frame intervals
        # which aren't integers...this adds a little imprecision but is likely
        # good enough
        frameId = int(round(vidCapture.get(cv2.CAP_PROP_POS_FRAMES)))

        success, image = vidCapture.read()

        if (image is None): continue
        if (len(group_frame[frameId]) != 0):
            for idx, obj in enumerate(group_frame[frameId]):
                x, y, w, h = list(map(int, obj.coord))
                obj_id = obj.track_id
                save_path = os.path.join(train_path, f"{obj_id}")
                crop_obj = image[y:y + h, x:x + w]

                create_folder(save_path)
                save_crop_name = os.path.join(
                    save_path, "{}_{}_{}.jpg".format(cam_id, frameId,
                                                     duration_id))
                cv2.imwrite(save_crop_name, crop_obj)

    vidCapture.release()
    print("Complete folder {}".format(video_path))
    print()


def init_path(save_path):
    save_path = save_path
    train_path = os.path.join(save_path, "train")
    gallery_path = os.path.join(save_path, "gallery")
    query_path = os.path.join(save_path, "query")

    paths = [train_path, gallery_path, query_path]

    create_folder(save_path)
    for path in paths:
        create_folder(path)

    return paths


def create_data(save_path, data_dir, valid_scenarios=None):
    scenarios = glob.glob(os.path.join(data_dir, "*"))

    for scenario in scenarios:
        if scenario.split('/')[-1] not in valid_scenarios: continue

        print(scenario)
        durations = glob.glob(os.path.join(scenario, "videos/*"))
        # start_time = time.time()
        for duration in durations:
            if not os.path.isdir(duration): continue

            duration_id = duration.split('/')[-1]
            cameras = glob.glob(os.path.join(duration, "*"))
            for camera in cameras:
                if camera.split('/')[-1] == 'multiple_view.mp4': continue

                print(camera)
                video_path = camera
                camera_id = camera.split('/')[-1][:6]
                annotation_path = f"{scenario}/MOT_gt_processed_v2/{duration_id}/{camera_id}/gt/gt.txt"
                generate_frames(video_path, save_path, annotation_path)
        # end_time = time.time()
        # print('total time: ', (end_time - start_time) / 60)


def main(save_path, dataset_dir):

    train_valid_scenarios = ['1b', '8d', '8c']

    val_valid_scenarios = ['3b']

    # create dataset folder
    train_path, gallery_path, query_path = init_path(save_path)

    # create train data
    create_data(train_path, dataset_dir, train_valid_scenarios)

    # create validation data
    create_data(gallery_path, dataset_dir, val_valid_scenarios)

    # split data in validation data
    split_gallery_query(query_path, gallery_path)


if __name__ == "__main__":
    save_path = "data/vtx"
    dataset_dir = "../datasets/VTX/COMBINE_DATA_V3"
    print("START")
    main(save_path, dataset_dir)