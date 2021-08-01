import cupoch as cph

if __name__ == "__main__":
    camera_intrinsics = cph.camera.PinholeCameraIntrinsic(
        cph.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    kop = cph.kinfu.KinfuOption()
    kop.distance_threshold = 5.0
    kinfu = cph.kinfu.KinfuPipeline(camera_intrinsics, kop)

    for i in range(5):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = cph.io.read_image(
            "../../testdata/rgbd/color/{:05d}.jpg".format(i))
        depth = cph.io.read_image(
            "../../testdata/rgbd/depth/{:05d}.png".format(i))
        rgbd = cph.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        res = kinfu.process_frame(rgbd)
        if res:
            print(kinfu.cur_pose)

    print("Extract triangle mesh")
    mesh = kinfu.extract_triangle_mesh()
    cph.visualization.draw_geometries([mesh])
