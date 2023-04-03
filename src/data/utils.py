from collections import defaultdict


class Human():

    def __init__(self, frame, track_id, coord, frame_start) -> None:
        self.frame = int(frame) - frame_start
        self.track_id = int(track_id)
        self.coord = list(map(float, coord))
        self.area = self.get_area()

    def get_area(self):
        return self.coord[2] * self.coord[3]

    def xywh_to_xyxy(self):
        x, y, w, h = self.coord
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        return (xmin, ymin, xmax, ymax)


def get_object_frame(file, frame_start):
    with open(file, "rt") as f:
        lines = f.readlines()

    lines = list(map(lambda x: x.strip().split(",")[:6], lines))

    humans = []
    for line in lines:
        humans.append(Human(line[0], line[1], line[2:], frame_start))

    group_frame = defaultdict(list)
    for human in humans:
        group_frame[human.frame].append(human)
    return group_frame