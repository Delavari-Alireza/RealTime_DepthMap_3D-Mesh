import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import argparse
import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using Depth Anything V2 for video')

    parser.add_argument('--video-path', type=str, default=0)

    # parser.add_argument('--outdir', type=str, default='./vis_depth')

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device_map='cuda')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)

    frame_width = int(cap.get(3)) * 2
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    print(size)

    width, height = (int(cap.get(3)), int(cap.get(4)))

    pinholeCamera = o3d.camera.PinholeCameraIntrinsic(width, height, 470.5, 470.5, width / 2, height / 2)

    frame_count = 0

    first_time = True
    while cap.isOpened():
        ret, raw_image = cap.read()
        # print(size)
        if not ret:
            break

        # print()

        frame_count += 1

        # if args.video_path != '0':
        #     # print("ok")
        #     if frame_count % 10 != 0:

        #         continue

        # frame_count += 1

        pil_image = Image.fromarray(raw_image)

        depth = pipe(pil_image)["depth"]

        depth = np.array(depth)
        depth = -depth

        depth_gray = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        tmp1 = o3d.geometry.Image(raw_image.astype(np.uint8))
        tmp2 = o3d.geometry.Image(depth_gray.astype(np.uint8))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(tmp1, tmp2, depth_scale=1.0,
                                                                        convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinholeCamera)

        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        # rgbd_image,
        # o3d.camera.PinholeCameraIntrinsic(
        #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])

        if (first_time):
            vis.add_geometry(pcd)
            last_pcd = pcd
            first_time = False
        else:

            vis.add_geometry(pcd, reset_bounding_box=False)
            vis.remove_geometry(last_pcd, reset_bounding_box=False)
            # vis.remove_geometry( last_pcd, reset_bounding_box=True)
            # vis.add_geometry(pcd)
            last_pcd = pcd
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        # resized_pred = Image.fromarray(depth).resize((width, height), Image.NEAREST)

        # x, y = np.meshgrid(np.arange(width), np.arange(height))
        # x = (x - width / 2) / 400#470.5
        # y = (y - height / 2) / 400#470.5
        # z = np.array(resized_pred)
        # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        # colors = raw_image.reshape(-1, 3)#.reshape(-1, 3) / 255.0#np.array(pil_image).reshape(-1, 3) / 255.0

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        # # print( raw_image.shape  , depth.shape )

        combined_results = cv2.hconcat([cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB), depth_gray])

        # o3d.visualization.draw_geometries([pcd])
        # out_video.write(combined_results)

        cv2.imshow("result", combined_results)

        # Press q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # vis.destroy_window()
            break

    vis.destroy_window()
    cap.release()
    cv2.destroyAllWindows()



