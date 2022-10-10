import open3d as o3d
import meshio
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

save_filepath = "../../train/0422_graphAE_dfaust/test_30/images_1/"
read_filepath = "../../train/0422_graphAE_dfaust/test_30/epoch198/ply"


# adapted from https://nbviewer.org/github/empet/Hollow-mask-illusion/blob/main/Hollow-Mask-illusion-Animation.ipynb

def scene_to_png(scene, png_filename):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # scale.
    obb = scene
    # center.
    mesh = obb
    o3d.io.write_triangle_mesh("sample.obj", mesh)

    msh = meshio.read("sample.obj")
    verts = msh.points[:, :3]
    I, J, K = msh.cells_dict["triangle"].T
    x, z, y = verts.T

    fig = make_subplots(rows=1, cols=1,
                        horizontal_spacing=0.015,
                        specs=[[{'type': 'scene'}]])

    colorscale = [[0, 'rgb(100,100,100)'],
                  [1, 'rgb(250,250,250)']]
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                            i=I, j=J, k=K,
                            intensity=z,
                            colorscale=colorscale,
                            showscale=False,
                            lighting=dict(ambient=0.1,
                                          diffuse=1,
                                          fresnel=3,
                                          specular=0.5,
                                          roughness=0.05),
                            lightposition=dict(x=100,
                                               y=200,
                                               z=1000)
                            ), 1, 1)
    axis_prop = dict(visible=False, autorange=False)

    plotly_scenes = dict(xaxis=dict(range=[-11.41, 11.41], **axis_prop),
                         yaxis=dict(range=[-11.41, 11.41], **axis_prop),
                         zaxis=dict(range=[-11.41, 11.41], **axis_prop),
                         camera_eye=dict(x=1.85, y=1.85, z=0.65),
                         aspectratio=dict(x=10, y=10, z=10),
                         )  # annotations = bbox_annotations)
    fig.update_layout(title_text=f"Generation for scene 0", title_x=0.5, title_y=0.95,
                      font_size=12, font_color="white",
                      width=800, height=400, autosize=False,
                      margin=dict(t=2, r=2, b=2, l=2),
                      paper_bgcolor='black',
                      scene=plotly_scenes)
    fig.write_image(png_filename)

for i in range(1, 1000):
    if i <= 9:
        j = "/0000000"
    elif i <= 99:
        j = "/000000"
    else:
        j = "/00000"
    read = read_filepath + j + str(i) + "_out.ply"
    write = save_filepath + str(i) + ".png"
    img = o3d.io.read_triangle_mesh(read)
    scene_to_png(img, write)

#to convert to gif, convert -delay 2 {1..99}.png -delay 100 100.png -dispose Background human_1.gif



