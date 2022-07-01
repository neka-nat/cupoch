import cupoch as cph

if __name__ == "__main__":

    print("Testing IO for point cloud ...")
    pcd = cph.io.read_point_cloud("../../testdata/fragment.pcd")
    print(pcd)
    cph.io.write_point_cloud("copy_of_fragment.pcd", pcd)

    print("Testing IO for meshes ...")
    mesh = cph.io.read_triangle_mesh("../../testdata/knot.ply")
    print(mesh)
    cph.io.write_triangle_mesh("copy_of_knot.ply", mesh)

    print("Testing IO for textured meshes ...")
    textured_mesh = cph.io.read_triangle_mesh("../../testdata/crate/crate.obj")
    print(textured_mesh)
    cph.io.write_triangle_mesh("copy_of_crate.obj", textured_mesh, write_triangle_uvs=True)
    copy_textured_mesh = cph.io.read_triangle_mesh("copy_of_crate.obj")
    print(copy_textured_mesh)

    print("Testing IO for images ...")
    img = cph.io.read_image("../../testdata/lena_color.jpg")
    print(img)
    cph.io.write_image("copy_of_lena_color.jpg", img)
