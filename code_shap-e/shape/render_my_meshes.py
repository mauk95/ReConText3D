# ==============================================================================
# Modified render script for custom mesh directory
# Based on render_script_shapE.py
# ==============================================================================

import bpy
import sys
import mathutils
from mathutils import Vector, Matrix, Euler
import argparse
import numpy as np
import math
import os
import time
import pickle
from PIL import Image
import random
from IPython import embed

## solve the division problem
from decimal import Decimal, getcontext
getcontext().prec = 28  # Set the precision for the decimal calculations.

parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", type = str, required=True, help="Directory containing PLY mesh files")
parser.add_argument("--save_dir", type = str, default='./rendering_output', help="Directory to save rendered images")

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

# Get specific PLY file for testing
test_ply = "0a35ed754a996ca2f8512e444fc30da8d05d393bae28218e23e2e3749ed40e5b.ply"
uid_paths = [os.path.join(args.parent_dir, test_ply)]
os.makedirs(args.save_dir, exist_ok=True)

print(f"Found {len(uid_paths)} PLY files to render")

bpy.context.scene.render.engine = 'CYCLES'
# small samples for fast rendering
bpy.context.scene.cycles.samples = 16
# bpy.context.scene.cycles.samples = 128
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

# Initialize depth rendering (matching render.py approach)
def init_nodes(save_depth=True):
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    
    # Enable depth pass
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = save_depth
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    
    # Clear existing nodes
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        
        # Map range node to remap depth values (matching original render.py)
        depth_map = nodes.new(type="CompositorNodeMapRange")
        depth_map.inputs[1].default_value = 0  # min value
        depth_map.inputs[2].default_value = 10  # max value
        depth_map.inputs[3].default_value = 0  # min output (black object)
        depth_map.inputs[4].default_value = 1  # max output (white background)
        
        links.new(render_layers.outputs['Depth'], depth_map.inputs[0])
        links.new(depth_map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = depth_map
    
    return outputs, spec_nodes

# Initialize depth nodes
outputs, spec_nodes = init_nodes(save_depth=True)

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    if 'NVIDIA' in d['name']:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    else:
        d["use"] = 0 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

render_prefs = bpy.context.preferences.addons['cycles'].preferences
render_device_type = render_prefs.compute_device_type
compute_device_type = render_prefs.devices[0].type if len(render_prefs.devices) > 0 else None
# Check if the compute device type is GPU
if render_device_type == 'CUDA' and compute_device_type == 'CUDA':
    # GPU is being used for rendering
    print("Using GPU for rendering")
else:
    # GPU is not being used for rendering
    print("Not using GPU for rendering")


# if the object is too far away from the origin, pull it closer
def check_object_location(mesh_objects, max_distance):
    # Compute the maximum distance of any object from the origin
    max_obj_distance = max(obj.location.length for obj in mesh_objects)

    # If any object is too far from the origin, move all mesh_objects closer to the origin
    if max_obj_distance > max_distance:
        print("mesh_objects are too far from the origin. Centering objects...")
        bbox_center, _ = compute_bounding_box(mesh_objects)
        for obj in mesh_objects:
            obj.location -= bbox_center
        bpy.context.view_layer.update()
    else:
        print("Object initial locations are within range.")
        print('max_obj_distance:', max_obj_distance, 'max_distance:', max_distance)

    # Compute the maximum distance again and check if it's within range
    max_obj_distance = max(obj.location.length for obj in mesh_objects)
    if max_obj_distance > max_distance:
        print("Objects are still too far from the origin. Please adjust the object locations and try again.")
        return False
    else:
        print("Object locations are within range.")
        return True

# compute the bounding box of the mesh objects
def compute_bounding_box(mesh_objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in mesh_objects:
        matrix_world = obj.matrix_world
        mesh = obj.data

        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co

            min_coords = Vector((min(min_coords[i], global_coord[i]) for i in range(3)))
            max_coords = Vector((max(max_coords[i], global_coord[i]) for i in range(3)))

    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    return bbox_center, bbox_size

# normalize objects 
def normalize_and_center_objects(mesh_objects, normalization_range):

    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    # Check the location of the objects and move them closer to the origin if necessary
    check_object_location(mesh_objects, 1000)

    # Compute the bounding box of the objects again after making adjustments
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    # Normalize the objects within a certain range
    max_dimension = max(bbox_size.x, bbox_size.y, bbox_size.z)
    scaling_factor = normalization_range / max_dimension

    for obj in mesh_objects:
        mesh = obj.data
        matrix_world = obj.matrix_world
        inv_matrix_world = matrix_world.inverted()
        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co
            global_coord -= bbox_center
            global_coord *= scaling_factor
            vert.co = inv_matrix_world @ global_coord
        mesh.update()
        obj.data.update()

    bpy.context.view_layer.update()
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)
    print('final bbox_center: ', bbox_center)
    print('final bbox_size: ', bbox_size)

    return bbox_center, bbox_size

# check if rendered object will cross the boundary of the image
def project_points_to_camera_space(obj, camera):
    bpy.context.view_layer.update()
    # Get the 8 corners of the bounding box in local space
    bbox_local = [Vector(corner) for corner in obj.bound_box]
    # print(bbox_local)

    # Transform bounding box corners to world space
    bbox_world = [obj.matrix_world @ corner for corner in bbox_local]
    bbox_world = [np.array(corner) for corner in bbox_world]  # convert to numpy

    # Get the 4x4 transformation matrix of the camera
    RT = np.array(camera.matrix_world.inverted())
    RT = RT[:3, :4]  # Remove the last row to make it a 3x4 matrix

    # Get the intrinsic matrix K from the camera properties
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    f_x = width / 2.0 / np.tan(camera.data.angle / 2.0)
    f_y = height / 2.0 / np.tan(camera.data.angle / 2.0)
    c_x = width / 2.0
    c_y = height / 2.0

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    bbox_camera = []
    bbox_image = []

    for vertex in bbox_world:
        # Transform from world to camera space
        XYZ_camera = np.dot(RT, np.append(vertex, 1))  # Append 1 to make it a 4-element vector for multiplication with RT

        # Project from camera space to image space
        XYZ_image = np.dot(K, XYZ_camera)

        # Homogenize to get pixel coordinates
        XYZ_image /= XYZ_image[2]

        bbox_camera.append(XYZ_camera)
        bbox_image.append(XYZ_image[:2])  # Keep only x and y

    # Check if the coordinates are within the normalized device coordinates [-1, 1]
    is_within_ndc = all(np.all(np.abs(vertex[:2]) <= 1) for vertex in bbox_image)

    # print(is_within_ndc)
    return bbox_image

# prepare the scene
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Create lights

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()

def create_light(name, light_type, energy, location, rotation):
    bpy.ops.object.light_add(type=light_type, align='WORLD', location=location, scale=(1, 1, 1))
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.rotation_euler = rotation
    return light

def three_point_lighting():
    
    # Key light
    key_light = create_light(
        name="KeyLight",
        light_type='AREA',
        energy=1000,
        location=(4, -4, 4),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    key_light.data.size = 2

    # Fill light
    fill_light = create_light(
        name="FillLight",
        light_type='AREA',
        energy=300,
        location=(-4, -4, 2),
        rotation=(math.radians(45), 0, math.radians(135))
    )
    fill_light.data.size = 2

    # Rim/Back light
    rim_light = create_light(
        name="RimLight",
        light_type='AREA',
        energy=600,
        location=(0, 4, 0),
        rotation=(math.radians(45), 0, math.radians(225))
    )
    rim_light.data.size = 2

three_point_lighting()

os.makedirs(os.path.join(args.save_dir, 'Cap3D_imgs'), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'depth'), exist_ok=True)
for i in range(4):
    os.makedirs(os.path.join(args.save_dir, 'Cap3D_cams', 'Cap3D_imgs_view%d_CamMatrix'%i), exist_ok=True)

def load_ply(filepath):
    import plyfile
    plydata = plyfile.PlyData.read(filepath)
    
    verts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    faces = np.vstack(plydata['face']['vertex_index'])
    vertex_colors = np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T / 255

    mesh = bpy.data.meshes.new(name="Imported PLY")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())

    # create color layer
    color_layer = mesh.vertex_colors.new()

    # assign colors to vertices
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            loop_vert_index = mesh.loops[loop_index].vertex_index
            color_layer.data[loop_index].color = vertex_colors[loop_vert_index].tolist() + [1.0]

    # create new material
    mat = bpy.data.materials.new(name="VertexCol")
    
    # enable 'use_nodes'
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # get the 'Material Output' node
    material_output = nodes.get('Material Output')
    
    # add 'Vertex Color' node
    vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
    
    # add 'BSDF' node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # link 'Vertex Color' node to 'BSDF' node
    mat.node_tree.links.new(vertex_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    
    # link 'BSDF' node to 'Material Output' node
    mat.node_tree.links.new(bsdf_node.outputs['BSDF'], material_output.inputs['Surface'])

    # Create new object and link mesh and material
    obj = bpy.data.objects.new("ImportedPLY", mesh)
    obj.data.materials.append(mat)

    # Link object to the current collection
    bpy.context.collection.objects.link(obj)

    return mesh

for uid_path in uid_paths:
    if not os.path.exists(uid_path):
        continue

    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    #bpy.ops.import_mesh.ply(filepath=uid_path)
    from bpy_extras.object_utils import object_data_add
    mesh = load_ply(uid_path)
    obj_added = object_data_add(bpy.context, mesh)

    print('begin*************')
    
    # Normalize scene (matching render.py approach)
    def scene_bbox():
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene():
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = scene_bbox()
        scale = 1 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale

        bpy.context.view_layer.update()
        bbox_min, bbox_max = scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        
        return scale, offset

    # Normalize the scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')


    # Use existing camera (keep original orientation)
    camera = bpy.context.scene.camera
    name = uid_path.split('/')[-1].split('.')[0]
    
    # 4 views: yaw = 0°, 90°, 180°, 270°, pitch = 30°, radius = 3.0, fov = 0.691111 (more zoomed out)
    views = [
        {'yaw': 0, 'pitch': math.radians(30), 'radius': 3.0, 'fov': 0.691111},
        {'yaw': math.radians(90), 'pitch': math.radians(30), 'radius': 3.0, 'fov': 0.691111},
        {'yaw': math.radians(180), 'pitch': math.radians(30), 'radius': 3.0, 'fov': 0.691111},
        {'yaw': math.radians(270), 'pitch': math.radians(30), 'radius': 3.0, 'fov': 0.691111}
    ]
    
    for camera_opt in range(-1, 4):
        # use transparent background to adjust camera distance
        if camera_opt == -1:
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.scene.render.film_transparent = True
            # Use first view for background test
            view = views[0]
            camera.location = Vector((
                view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['pitch'])
            ))
            camera.data.lens = 16 / np.tan(view['fov'] / 2)
        elif camera_opt == 0:
            img_path = os.path.join(args.save_dir, 'Cap3D_imgs', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
            img = Image.open(img_path)
            img_array = np.array(img)
            if np.sum(img_array<10) > 1020000:
                print(name, 'WARNING: rendered image may contain too much white space')

            # change to black background to render the final views (matching render.py)
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
            bpy.context.scene.render.film_transparent = False
            # Use first view for background test
            view = views[0]
            camera.location = Vector((
                view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['pitch'])
            ))
            camera.data.lens = 16 / np.tan(view['fov'] / 2)
            
            # Update depth map range based on camera distance (matching original render.py)
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)

            # No need for bounding box checking with render.py approach
        else:
            # Use specific view for views 1, 2, 3
            view = views[camera_opt]
            camera.location = Vector((
                view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['pitch'])
            ))
            camera.data.lens = 16 / np.tan(view['fov'] / 2)
            
            # Update depth map range based on camera distance (matching original render.py)
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)

        # Make the camera point at the origin (restore original behavior)
        direction = (Vector((0, 0, 0)) - camera.location).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()

        camera.data.clip_start = 0.1
        camera.data.clip_end = max(1000, view['radius'] * 2)

        print('camera.location: ', camera.location)

        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        if camera_opt == -1:
            file_path = os.path.join(args.save_dir, 'Cap3D_imgs', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
            bpy.context.scene.render.filepath = file_path
            if os.path.exists(file_path):
               continue
        else:
            file_path = os.path.join(args.save_dir, 'Cap3D_imgs', '%s_%d.png'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
        bpy.context.scene.render.filepath = file_path
        
        # Set depth output path (matching render.py approach)
        outputs['depth'].file_slots[0].path = os.path.join(args.save_dir, 'depth', '%s_%d_depth'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
        
        if os.path.exists(file_path):
           continue

        bpy.ops.render.render(write_still=True)
        
        # Rename depth file to proper extension (matching render.py)
        import glob
        depth_files = glob.glob(f'{outputs["depth"].file_slots[0].path}*.png')
        if depth_files:
            os.rename(depth_files[0], f'{outputs["depth"].file_slots[0].path}.png')
        
        # Delete bg_img (used for setting scale) when rendering is done
        bg_img = os.path.join(args.save_dir, 'Cap3D_imgs', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
        if camera_opt == 3 and os.path.exists(bg_img):
            os.remove(bg_img)

        def get_3x4_RT_matrix_from_blender(cam):
            # Use matrix_world instead to account for all constraints
            location, rotation = cam.matrix_world.decompose()[0:2]
            R_world2bcam = rotation.to_matrix().transposed()

            # Use location from matrix_world to account for constraints:     
            T_world2bcam = -1*R_world2bcam @ location

            # put into 3x4 matrix
            RT = Matrix((
                R_world2bcam[0][:] + (T_world2bcam[0],),
                R_world2bcam[1][:] + (T_world2bcam[1],),
                R_world2bcam[2][:] + (T_world2bcam[2],)
                ))
            return RT

        if camera_opt>=0:
            RT = get_3x4_RT_matrix_from_blender(camera)
            
            RT_path = os.path.join(args.save_dir, 'Cap3D_cams', 'Cap3D_imgs_view%d_CamMatrix'%camera_opt, '%s_%d.npy'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            if os.path.exists(RT_path):
                continue
            np.save(RT_path, RT)

bpy.ops.wm.quit_blender()
