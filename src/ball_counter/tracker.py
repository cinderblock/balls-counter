"""Simple centroid tracker for associating detections across frames."""

from collections import OrderedDict

import numpy as np


class CentroidTracker:
    """
    Track objects by matching centroids frame-to-frame.

    Assigns a unique ID to each tracked object. When a centroid disappears
    for `max_disappeared` consecutive frames, it is deregistered.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float | None = None):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: np.ndarray) -> int:
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_id += 1
        return object_id

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids: list[tuple[int, int]]) -> OrderedDict[int, np.ndarray]:
        """
        Update tracked objects with new detections.

        Returns the current mapping of object ID -> centroid.
        """
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(centroids)

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute distance matrix between existing and new centroids
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :],
            axis=2,
        )

        # Greedy matching: closest pairs first
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if self.max_distance is not None and distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in set(range(len(object_centroids))) - used_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Register new unmatched detections
        for col in set(range(len(input_centroids))) - used_cols:
            self.register(input_centroids[col])

        return self.objects
