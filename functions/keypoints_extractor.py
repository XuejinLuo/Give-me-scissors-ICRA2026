# feature_extractor.py
import numpy as np
import open3d as o3d
import cv2
import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, DBSCAN
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from functions.utils import pca_transform

class KeypointsExtractor(object):
    def __init__(self, config, origin_image_path, roi_image_path, save_path):
        self.config = config
        self.bounds_min = np.array(self.config['main']['bounds_min'])
        self.bounds_max = np.array(self.config['main']['bounds_max'])
        self.origin_image_path = origin_image_path
        self.roi_image_path = roi_image_path
        self.save_path = save_path
        self.dinov2_vits14 = dinov2_vits14 = torch.hub.load('/home/luo/.cache/torch/hub/facebookresearch_dinov2_main', 
                                    'dinov2_vits14_reg', trust_repo=True, source='local').cuda()
        self.features_flat = None
        self.candidate_keypoints = None
        self.candidate_pixels = None
        self.candidate_rigid_group_ids = None
        self.projected = None

    def Get_Keypoints(self, point_cloud, masks):
        print("Get_Keypoints")
        self._Get_features()
        print("_Cluster_features")
        self.Show_features()
        self._Cluster_features(point_cloud, self.features_flat, masks)
        print("_Filter_points_by_bounds")
        self._Filter_points_by_bounds()
        print("_Merge_clusters")
        self._Merge_clusters()
        print("_Lexsort")
        self._Lexsort()
        # self._YOLO_BoundingBox()
        self._Map_points_to_original_image()
        self._Show_Projected_Image()
        return self.projected, self.candidate_keypoints, self.candidate_pixels, self.candidate_rigid_group_ids

    def _Get_features(self):
        image = cv2.imread(self.roi_image_path)
        height, width, channels = image.shape
        patch_size = 14
        patch_h = int(height // patch_size)
        patch_w = int(width // patch_size)
        new_height = patch_h * patch_size
        new_width = patch_w * patch_size
        transformed_rgb = cv2.resize(image, (new_width, new_height))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.
        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(device='cuda')  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self.dinov2_vits14.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(height, width),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim]
        self.features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # float32 [H*W, feature_dim]

    def _Cluster_features(self, point_cloud, features_flat, masks):
        #- start capture 1
        # height = self.global_config['camera']['roi_y_max']  - self.global_config['camera']['roi_y_min']
        # width = self.global_config['camera']['roi_x_max'] - self.global_config['camera']['roi_x_min']
        # points = np.zeros((height * width, 3), dtype=np.float32)
        # points[:, 0] = point_cloud['f0'].flatten()
        # points[:, 1] = point_cloud['f1'].flatten()
        # points[:, 2] = point_cloud['f2'].flatten()

        #- start capture 2
        points = point_cloud.transpose(1, 2, 0).reshape(-1, 3)

        points_tensor = torch.tensor(points).cuda()
        features_flat_tensor = torch.tensor(features_flat).cuda()
        masks_tensor = [torch.tensor(mask["segmentation"]).cuda() for mask in masks]

        self.candidate_keypoints, self.candidate_pixels, self.candidate_rigid_group_ids = self.cluster_features(
            points_tensor,
            features_flat_tensor,
            masks_tensor
        )

    def _Filter_points_by_bounds(self):
        bounds_min = self.bounds_min.copy()
        bounds_max = self.bounds_max.copy()
        within_space = self.filter_points_by_bounds(self.candidate_keypoints, bounds_min, bounds_max, strict=True)
        self.candidate_keypoints = self.candidate_keypoints[within_space]
        self.candidate_pixels = self.candidate_pixels[within_space]
        self.candidate_rigid_group_ids = self.candidate_rigid_group_ids[within_space]

    def _Merge_clusters(self):
        merged_indices = self.merge_clusters_DBSCAN(self.candidate_keypoints)
        self.candidate_keypoints = self.candidate_keypoints[merged_indices]
        self.candidate_pixels = self.candidate_pixels[merged_indices]
        self.candidate_rigid_group_ids = self.candidate_rigid_group_ids[merged_indices]

    def _Lexsort(self):
        sort_idx = np.lexsort((self.candidate_pixels[:, 0], self.candidate_pixels[:, 1]))
        self.candidate_keypoints = self.candidate_keypoints[sort_idx]
        self.candidate_pixels = self.candidate_pixels[sort_idx]
        self.candidate_rigid_group_ids = self.candidate_rigid_group_ids[sort_idx]

    def _YOLO_BoundingBox(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from ultralytics import YOLO
        MODEL_PATH = 'models/best_yolov8n.pt'
        IMAGE_PATH = self.roi_image_path
        CONFIDENCE_THRESHOLD = 0.5
        save_file_path = os.path.join(self.save_path, "yolo_roi_with_boxes.png")
        model = YOLO(MODEL_PATH)
        results = model.predict(source=IMAGE_PATH, conf=CONFIDENCE_THRESHOLD, save=False, show=False)
        self.yolo_boxes = []
        for i, r in enumerate(results):
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1_origin = x1 + self.config['camera']['roi_x_min']  # x-direction offset
                y1_origin = y1 + self.config['camera']['roi_y_min']
                x2_origin = x2 + self.config['camera']['roi_x_min']
                y2_origin = y2 + self.config['camera']['roi_y_min']

                self.yolo_boxes.append((x1, y1, x2, y2, class_name, confidence))
                # self.yolo_boxes.append((x1_origin, y1_origin, x2_origin, y2_origin, class_name, confidence))

    def _Map_points_to_original_image(self):
        """
        Map a set of points from ROI to original image coordinates.

        Parameters:
        points_in_roi (numpy.ndarray): Point coordinate array in ROI, shape (N, 2)
        roi_x_min (int): Left-upper corner x coordinate of ROI
        roi_y_min (int): Left-upper corner y coordinate of ROI

        Returns:
        numpy.ndarray: Point coordinate array in original image, shape (N, 2)
        \"\"\"
        # Add ROI left-upper corner coordinates to each point
        self.points_in_roi = self.candidate_pixels
        roi_x_min = self.config['camera']['roi_x_min']
        roi_y_min = self.config['camera']['roi_y_min']
        self.candidate_pixels = self.points_in_roi + np.array([roi_y_min, roi_x_min])

    def _Show_Projected_Image(self):
        # self.projected = self.project_keypoints_to_image(self.origin_image_path, self.candidate_pixels) # whole image
        self.projected = self.project_keypoints_to_image(self.roi_image_path, self.points_in_roi) # roi image
        plt.imshow(self.projected)
        plt.axis('off')
        plt.show()
        plt.imsave(os.path.join(self.save_path, "image_projected.png"), self.projected)

#----------------------------------------------#

    def cluster_features(self, points, features_flat, masks):
        """
        for each mask, cluster in feature space to get meaningful regions, and use their centers as keypoint candidates
        """
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        max_mask_ratio = 500
        for rigid_group_id, binary_mask in enumerate(masks):
            # ignore mask that is too large
            binary_mask = binary_mask.cpu().numpy()
            if np.mean(binary_mask) > max_mask_ratio:
                continue

            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            indices = feature_pixels[:, 0] * binary_mask.shape[1] + feature_pixels[:, 1]

            valid_indices = []
            y_indices = feature_pixels[:, 0]
            x_indices = feature_pixels[:, 1]
            points_flat = points[y_indices * binary_mask.shape[1] + x_indices]
            valid_mask = ~torch.all(points_flat == 0, dim=1)
            valid_indices = indices[valid_mask.cpu().numpy()]
            feature_points = points[valid_indices]
            obj_features_flat = obj_features_flat[valid_mask.cpu().numpy()]
            feature_pixels = feature_pixels[valid_mask.cpu().numpy()]

            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.float()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            print("features_pca", features_pca.size())
            if len(feature_points) == 0:
                    continue
            if features_pca.size(0) == 0 or features_pca.size(1) == 0:
                print("features_pca is empty!")
                continue
            else:
                features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            if feature_points_torch.size(0) == 0 or feature_points_torch.size(1) == 0:
                print("feature_points_torch is empty!")
                continue
            else:
                feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # cluster features to get meaningful regions
            num_samples = X.shape[0]
            num_clusters_plan = 10
            if num_samples < 50 or torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
                print("Input to K-means is not enough!")
                continue
            else:
                cluster_ids_x, cluster_centers = kmeans(
                    X=X,
                    num_clusters=num_clusters_plan,
                    distance='euclidean',
                    device='cuda',
                )

            cluster_centers = cluster_centers.to(device='cuda')
            for cluster_id in range(10):  # num_candidates_per_mask
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                if member_idx.sum() > 0:
                    member_points = feature_points[member_idx]
                    member_pixels = feature_pixels[member_idx]
                    member_features = features_pca[member_idx]

                    # Calculate distance and find nearest index
                    dist = torch.norm(member_features - cluster_center, dim=-1)
                    closest_idx = torch.argmin(dist)
                    candidate_keypoints.append(member_points[closest_idx].cpu().numpy())
                    candidate_pixels.append(member_pixels[closest_idx])
                    candidate_rigid_group_ids.append(rigid_group_id)
                else:
                    print(f"No members found for cluster {cluster_id}.")
                    continue

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)


        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def filter_points_by_bounds(self, candidate_keypoints, bounds_min, bounds_max, strict=True):
        """
        Filter points by taking only points within workspace bounds.
        exclude keypoints that are outside of the workspace.
        """
        assert candidate_keypoints.shape[1] == 3, "points must be (N, 3)"
        bounds_min = bounds_min.copy()
        bounds_max = bounds_max.copy()
        if not strict:
            bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
            bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
            bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
        within_bounds_mask = (
            (candidate_keypoints[:, 0] >= bounds_min[0])
            & (candidate_keypoints[:, 0] <= bounds_max[0])
            & (candidate_keypoints[:, 1] >= bounds_min[1])
            & (candidate_keypoints[:, 1] <= bounds_max[1])
            & (candidate_keypoints[:, 2] >= bounds_min[2])
            & (candidate_keypoints[:, 2] <= bounds_max[2])
        )
        return within_bounds_mask

    def merge_clusters(self, candidate_keypoints):
        """
        merge close points by clustering in cartesian space    
        """
        mean_shift = MeanShift(bandwidth=0.06, bin_seeding=True, n_jobs=32)##min_dist_bt_keypoints
        mean_shift.fit(candidate_keypoints)
        cluster_centers = mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices

    def merge_clusters_DBSCAN(self, candidate_keypoints):
        """
        Merge close points by clustering in Cartesian space using DBSCAN.
        Smaller eps values lead to smaller clusters and may treat many points as noise.
        Larger eps values may lead to multiple different clusters being merged into one cluster.
        Smaller min_samples values make it easier to form clusters, but may also result in more noise points.
        Larger min_samples values require more points to form valid clusters, potentially excluding small aggregations as noise.
        """
        dbscan = DBSCAN(eps=0.025, min_samples=2) # surgery tool
        # dbscan = DBSCAN(eps=0.05, min_samples=3) # dualarm auto grasp

        labels = dbscan.fit_predict(candidate_keypoints)

        merged_indices = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # -1 indicates noise points, no need to aggregate
            
            cluster_points = candidate_keypoints[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            
            dist = np.linalg.norm(candidate_keypoints - cluster_center, axis=-1)
            merged_indices.append(np.argmin(dist))

        return merged_indices

    def sort_candidates(self, candidate_pixels):
        """
        sort candidates by locations 
        """    
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        return sort_idx

    # def project_keypoints_to_image(self, img_path, candidate_pixels):
    #     """
    #     project keypoints to image space
    #     """
    #     projected = cv2.imread(img_path)
    #     projected = cv2.cvtColor(projected, cv2.COLOR_BGR2RGB)

    #     if self.yolo_boxes is not None:
    #         for (x1, y1, x2, y2, class_name, confidence) in self.yolo_boxes:
    #             text = f"{class_name}"
    #             cv2.rectangle(projected, (x1, y1), (x2, y2), (255, 0, 0), 1)
    #             plt.text(x2, y1+5, f"{class_name}", 
    #                  color='r', fontsize=15)
                
    #     # overlay keypoints on the image
    #     for keypoint_count, pixel in enumerate(candidate_pixels):
    #         displayed_text = f"{keypoint_count}"
    #         text_length = len(displayed_text)
            
    #         # # draw a box
    #         # box_width = 9 + 6 * (text_length - 1)
    #         # box_height = 11
    #         # cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), 
    #         #             (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
    #         # cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), 
    #         #             (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 1)
            
    #         # draw text
    #         org = (pixel[1] - 4 * (text_length), pixel[0] + 4)
    #         color = (255, 0, 0)
    #         cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    #         keypoint_count += 1

    #     return projected
    
    def project_keypoints_to_image(self, img_path, candidate_pixels):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import numpy as np
        import os

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        fig, ax = plt.subplots(1, figsize=(img_array.shape[1]/100, img_array.shape[0]/100), dpi=100)
        ax.imshow(img_array)
        ax.axis('off')
        
        for keypoint_count, pixel in enumerate(candidate_pixels):
            x, y = pixel[1], pixel[0]
            
            ax.text(x - 4 * len(str(keypoint_count)), y + 4, 
                    str(keypoint_count), color='red', 
                    fontsize=8)
        
        plt.tight_layout(pad=0)
        
        fig.canvas.draw()
        projected = np.array(fig.canvas.renderer._renderer)
        plt.close(fig)
        
        if self.save_path:
            save_path = os.path.join(self.save_path, "image_projected.png")
            plt.imsave(save_path, projected)
        
        return projected

    def Show_features(self):
        ### Extract features
        image = cv2.imread(self.roi_image_path)
        height, width, channels = image.shape
        patch_size = 14
        patch_h = int(height // patch_size)
        patch_w = int(width // patch_size)
        feat_dim = 384
        features = self.extract_features(self.roi_image_path, patch_h, patch_w, feat_dim)
        features = features.reshape(4 * patch_h * patch_w, feat_dim)
        pca_features_rgb, pca_features, pca = pca_transform(features, patch_h, patch_w)
        print("height:", height)
        print("width:", width)
        plt.figure(figsize=(10, 10))
        plt.imshow(pca_features_rgb[0][...,::-1])
        plt.axis('off')
        plt.savefig(os.path.join(self.save_path, "image_features.png"))
        plt.show()

    def extract_features(self, img_path, patch_h, patch_w, feat_dim):
        transform = T.Compose([
            T.GaussianBlur(9, sigma=(0.1, 2.0)),  # Gaussian blur
            T.Resize((patch_h * 14, patch_w * 14)),  # Resize image
            T.CenterCrop((patch_h * 14, patch_w * 14)),  # Center crop
            T.ToTensor(),  # Convert to tensor
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalization
        ])

        dinov2_vits14 = torch.hub.load('/home/luo/.cache/torch/hub/facebookresearch_dinov2_main', 
                                    'dinov2_vits14_reg', trust_repo=True, source='local').cuda()
        # dinov2_vits14 = torch.hub.load('/home/luo/.cache/torch/hub/facebookresearch_dinov2_main', 
        #                                'dinov2_vits14', trust_repo=True, source='local').cuda()
        
        features = torch.zeros(4, patch_h * patch_w, feat_dim)
        img = Image.open(img_path).convert('RGB')
        imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()
        imgs_tensor[0] = transform(img)[:3]

        with torch.no_grad():
            features_dict = dinov2_vits14.forward_features(imgs_tensor)
            features = features_dict['x_norm_patchtokens']


        return features.cpu()

    def visualize_mask(self, binary_mask):
        plt.figure(figsize=(5, 5))
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Binary Mask')
        plt.axis('off')
        plt.show()