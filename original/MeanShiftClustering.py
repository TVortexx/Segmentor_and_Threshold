import cv2
import numpy as np


class MeanShiftClusterer:
    def __init__(self, image, bandwidth=100, max_iter=300, tol=1e-3, with_spatial_coords=False, use_luv_space=True):
        self.bandwidth = bandwidth
        self.merge_threshold = bandwidth / 2
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.image = image
        self.labels_ = None
        self.features = None
        self.shuffled_features = None
        self.use_luv_space = use_luv_space
        self.__initialize_features(with_spatial_coords)

    def __initialize_features(self, with_spatial_coords):
        h, w, c = self.image.shape
        if self.use_luv_space:
            luv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2LUV)
            features = luv_image.reshape(-1, c)
        else:
            features = self.image.reshape(-1, c)

        if with_spatial_coords:
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            spatial_features = np.dstack((x_coords, y_coords)).reshape(-1, 2)
            features = np.hstack((features, spatial_features))
        self.features = features

    def __mean_shift_single_center(self, start_idx):
        center = self.shuffled_features[start_idx]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(self.shuffled_features - center, axis=1)
            in_bandwidth = self.shuffled_features[distances < self.bandwidth]
            new_center = np.mean(in_bandwidth, axis=0)
            if np.linalg.norm(new_center - center) < self.tol:
                break
            center = new_center
        return center

    def __assign_labels(self):
        distances = np.linalg.norm(self.features[:, np.newaxis] - self.cluster_centers_, axis=2)
        self.labels_ = np.argmin(distances, axis=1).reshape(self.image.shape[0], self.image.shape[1])

    def get_clustered_image(self):
        if self.labels_ is None:
            raise ValueError("Labels have not been computed. Please call fit() first.")
        clustered_image = np.zeros_like(self.image)
        for i, center in enumerate(self.cluster_centers_):
            if self.use_luv_space:
                # transform center color back to rgb
                luv_color = np.array([[center[:3]]], dtype=np.uint8)  # shape (1,1,3)
                rgb_color = cv2.cvtColor(luv_color, cv2.COLOR_LUV2RGB)[0, 0]
            else:
                rgb_color = center[:3].astype(np.uint8)
            clustered_image[self.labels_ == i] = rgb_color
        return clustered_image

    def cluster(self):
        self.shuffled_features = np.random.permutation(self.features)
        centers = []
        visited = np.zeros(len(self.shuffled_features), dtype=bool)
        last_cluster_addition = 0
        for i in range(len(self.shuffled_features)):
            if i - last_cluster_addition >= 100:
                break

            if visited[i]:
                continue

            center = self.__mean_shift_single_center(i)

            distances = np.linalg.norm(self.shuffled_features - center, axis=1)
            visited[distances < self.merge_threshold] = True

            if any(np.linalg.norm(center - c) < self.merge_threshold for c in centers):
                continue

            centers.append(center)
            last_cluster_addition = i
            print(f"Iteration {i + 1}/{len(self.shuffled_features)}")
            print(f"Current center: {center}")
            print(f"Samples left: {np.sum(~visited)}")
            print(f"Centers found: {len(centers)}")

        self.cluster_centers_ = np.array(centers)
        self.__assign_labels()


if __name__ == "__main__":
    import cv2 as cv

    image = cv.imread('segment_images/school-entrance.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    clusterer = MeanShiftClusterer(image, bandwidth=100, with_spatial_coords=True)
    clusterer.cluster()

    clustered_image = clusterer.get_clustered_image()
    clustered_image = cv.cvtColor(clustered_image, cv.COLOR_RGB2BGR)
    cv.imshow('Clustered Image', clustered_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
