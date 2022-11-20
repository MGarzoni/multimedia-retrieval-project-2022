import os
from typing import List


class Database:
    shapeCount = 0
    shapes = {}

    def __init__(self, path: str, extensions: List[int]):
        self.path = path

        for folder in os.scandir(path):
            if not folder.is_dir():
                continue

            class_name = os.path.basename(os.path.normpath(folder))
            self.shapes[class_name] = []

            for file in os.scandir(folder):
                if not os.path.splitext(file)[-1].lower() in extensions:
                    continue

                self.shapeCount += 1
                self.shapes[class_name].append(os.path.basename(file))

    def __iter__(self):
        for className, files in self.shapes.items():
            for filename in files:
                yield className, filename

    def __len__(self):
        return self.shapeCount
