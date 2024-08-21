import trimesh
import open3d as o3d
# import urdfpy
import yourdfpy
import numpy as np
import os
import shutil
from functools import partial
from itertools import chain
import copy
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import textwrap
import re
import glob


def simplify_meshes(save_dir):
    src_dir = save_dir + '/meshes'
    dest_dir = save_dir + '/simple_meshes'

    decimation = 100 # reduce triangles by this factor

    src_ext = "stl"
    # dest_ext = "stl"

    # Compile a case-insensitive regular expression pattern for the extension
    pattern = re.compile(re.escape(src_ext), re.IGNORECASE)

    # Use glob to recursively find all files in the folder and its subfolders
    paths = glob.glob(os.path.join(src_dir, '**'), recursive=True)

    # Filter the files based on the case-insensitive extension pattern
    paths = [file for file in paths if pattern.search(file)]

    # Calculate the relative path between the original path and the matching path
    relative_paths = [os.path.relpath(path, src_dir) for path in paths]

    # # Print the list of files
    # for path in relative_paths:
    #     print(path)
        

    for path, rel_path in zip(paths,relative_paths):
        
        print(f"loading {rel_path}")
        o3d_mesh = load_to_o3d_mesh(path)
        print(o3d_mesh)
        # print(f"watertight={o3d_mesh.is_watertight()}")
        # o3d.visualization.draw_geometries([o3d_mesh])
        mesh_export = o3d_mesh
        
        mesh_export = simplify_o3d_mesh(o3d_mesh,decimation)
        # export
        # export_path = f"{dest_dir}/{rel_path}.{dest_ext}"
        export_path = f"{dest_dir}/{rel_path}"
        
        
        # Get the folder path of the file path
        folder_path = os.path.dirname(export_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        o3d.io.write_triangle_mesh(export_path,mesh_export,write_vertex_normals=True)
        # o3d.visualization.draw_geometries([mesh_export])
        print(f"saved to \"{export_path}\"")
        file_size_bytes_original = os.path.getsize(path)
        file_size_bytes_export = os.path.getsize(export_path)
        print(f"reduction ratio={(1-file_size_bytes_export/file_size_bytes_original)*100:.1f}%")
        print("-"*10)

def fix_mesh(mesh_file):
    # Load your Trimesh mesh
    trimesh_mesh = trimesh.load(mesh_file)  

    # Repair the mesh
    trimesh.repair.fill_holes(trimesh_mesh)

    # Extract vertices and triangles
    vertices = trimesh_mesh.vertices.astype(np.float64)
    triangles = trimesh_mesh.faces

    # Create Open3D TriangleMesh
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute vertex normals (if needed)
    open3d_mesh.compute_vertex_normals()
    open3d_mesh.compute_triangle_normals()
    
    return open3d_mesh


def load_to_o3d_mesh(path:str):
    if path.endswith('.dae'): # dae mesh
        import collada
        c_mesh = collada.Collada(path)
        o3d_mesh = o3d.geometry.TriangleMesh()
        for geom in c_mesh.geometries:
            print(geom)
            for triset in geom.primitives:
                if not type(triset) is collada.triangleset.TriangleSet:
                    continue
                print(geom,triset)
            
                o3d_mesh += o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(triset.vertex), # vertices
                    o3d.utility.Vector3iVector(triset.vertex_index) # triangles
                    )
    else: # stl/obj
        o3d_mesh = o3d.io.read_triangle_mesh(path)
        
    if o3d_mesh.is_watertight()==False: # fix water tightness
        trimesh_mesh = trimesh.load(path) # Load your Trimesh mesh
        # Repair the mesh
        print("mesh is not water tight, tyring to rill holes")
        if trimesh.repair.fill_holes(trimesh_mesh) is False:
            raise(NotImplementedError("cannot fix broken mesh"))
        # Create Open3D TriangleMesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
        
    o3d_mesh.compute_triangle_normals()
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def simplify_o3d_mesh(o3d_mesh:o3d.geometry.TriangleMesh, decimation:int,maximum_error=0.0005):
    if decimation<=1:
        return o3d_mesh
    mesh_smp = o3d_mesh\
        .remove_unreferenced_vertices()\
        .remove_duplicated_vertices()\
        .remove_duplicated_triangles()\
        .remove_degenerate_triangles()\
        .remove_non_manifold_edges()\
        .merge_close_vertices(maximum_error)\
        .simplify_vertex_clustering(maximum_error)\
        .simplify_quadric_decimation(
            target_number_of_triangles=int(len(o3d_mesh.triangles)/decimation),
            maximum_error = maximum_error
            )
    mesh_smp.compute_vertex_normals()
    mesh_smp.compute_triangle_normals()
    print(mesh_smp)
    return mesh_smp



def urdf_to_graph(urdf: yourdfpy.URDF) -> nx.DiGraph:
    """
    Converts a URDF object into a networkx DiGraph object.

    Parameters:
    urdf : URDF object
        The URDF object to be converted.

    Returns:
    nx.DiGraph
        A directed graph representing the connections between joints in the URDF.
    """
    graph = nx.DiGraph()
    for joint_name, joint in urdf.joint_map.items():
        graph.add_edge(joint.parent, joint.child,
                       type=joint.type, name=joint_name)
    return graph


def plot_graph(graph: nx.DiGraph, figsize=(6, 10)):
    """
    plot the graph using networkx library
    Example:
        urdf = yourdfpy.URDF.load('../assets/urdf/a1/a1_minimum.urdf')
        G = urdf_to_graph(urdf)
        plot_graph(G)
    """
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot',args='-Grankdir=LR')
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color='lightblue', node_size=2000,node_shape='o')
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrows=True,
        arrowstyle='->', arrowsize=20,min_target_margin=30,min_source_margin=30)
    # wrap labels to fit in graph
    labels = {n: '\n'.join(textwrap.wrap(n, width=12)) for n in graph.nodes}
    nx.draw_networkx_labels(
        graph, pos,labels=labels, ax=ax, font_size=10, font_family='sans-serif')
    edge_labels = {(u, v): f"{graph[u][v]['name']}\n\n[{graph[u][v]['type']}]" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels, font_size=10,bbox=dict(alpha=0.0))
    plt.axis('off')  # Turn off axis labels
    # plt.title("urdf")
    return fig, ax


def collapse_edges(graph: nx.DiGraph, parent, attribute='type', match_value='fixed'):
    """
    Collapses edges in a graph where the specified attribute is equal to the match_value.

    Args:
        graph (networkx.Graph): The graph to collapse edges on.
        parent (object): The parent node to start collapsing from.
        attribute (str): The attribute to match edges on. Defaults to 'type'.
        match_value (object): The value the attribute should match to be collapsed. Defaults to 'fixed'.

    Returns:
        networkx.DiGraph: The graph with collapsed edges.
    """
    # Create a queue to traverse the graph
    frontiers = [parent]

    while len(frontiers) > 0:
        parent = frontiers.pop(0)

        # If the parent node is not in the graph, continue
        if parent not in graph:
            continue

        # Iterate over the children of the parent node
        for child in list(graph[parent]):
            # If the attribute of the child edge is equal to the match_value
            if graph[parent][child][attribute] == match_value:
                # Iterate over the grand children of the child node
                for grand_child in graph[child]:
                    # Add a new edge between the parent and grand child, resetting the attribute to match_value
                    graph.add_edge(parent, grand_child, **
                                   graph[child][grand_child])
                    frontiers.append(grand_child)
                # Remove the child node and its associated edges from the graph
                graph.remove_node(child)
                # Append the parent node to the queue to check again
                frontiers.append(parent)
            else:
                frontiers.append(child)

    # Return the graph with collapsed edges
    return graph


def get_leaf_nodes(urdf: yourdfpy.URDF, collapse_fixed_joints: bool = False) -> List[str]:
    """
    Returns a list of leaf nodes in a given URDF graph.

    Args:
        urdf: A yourdfpy.URDF object representing the URDF structure.
        collapse_fixed_joints: A boolean indicating whether to collapse fixed joints.

    Returns:
        List[str]: A list of leaf nodes in the URDF graph.
    """
    graph = urdf_to_graph(urdf)

    def get_leaf_nodes_helper(graph: nx.DiGraph) -> List[str]:
        return [node for node in graph.nodes() if graph.degree(node) == 1]

    if collapse_fixed_joints:
        collapsed_graph = collapse_edges(graph.copy(), urdf.base_link)
        return get_leaf_nodes_helper(collapsed_graph)
    else:
        return get_leaf_nodes_helper(graph)

# def trace_edges(graph: nx.DiGraph, start_node):
#     """trace the edges from a start node back to the root"""
#     edge_names = []
#     current_node = start_node
#     while True:
#         predecessors = tuple(graph.predecessors(current_node))
#         if not predecessors:  # No more parents, we've reached the root
#             break
#         parent_node = predecessors[0]
#         edge_data = graph.get_edge_data(parent_node, current_node)
#         if edge_data['type'] != 'fixed':
#             edge_names.append(edge_data['name'])
#         current_node = parent_node
#     return list(reversed(edge_names))

def trace_edges(graph: nx.DiGraph, start_node):
    """trace up the edges from a start node back to the root, then trace down from the same chain
    of the parent node to all leafs, return the edge names in a list
    """
    
    current_node = start_node
    candidate_node_chain = list()
    while True:
        predecessors = tuple(graph.predecessors(current_node))
        if not predecessors:  # No more parents, we've reached the root
            break
        candidate_node_chain.append(current_node)
        current_node = predecessors[0]
    if len(candidate_node_chain)==0:
        raise ValueError(f"trace_edges: start_node {start_node} does not have a parent!")
    if len(candidate_node_chain)==1: # only 1 node in the chain
        edge_data = graph.get_edge_data(current_node, candidate_node_chain[0])
        if edge_data['type'] != 'fixed':
            return [edge_data['name']]
        else:
            raise ValueError(f"trace_edges: start_node {start_node} dos not have a moving joint!")
    # reverse the list to get parent -> child
    candidate_node_chain = list(reversed(candidate_node_chain))

    # find the first node that is not fixed
    for i in range(len(candidate_node_chain)-1):
        edge_data = graph.get_edge_data(candidate_node_chain[i], candidate_node_chain[i+1])
        if edge_data['type'] != 'fixed':
            start_node = candidate_node_chain[i+1]
            edge_names = [edge_data['name']]
            # all_edges = [(candidate_node_chain[i], candidate_node_chain[i+1])]
            break

    
    # start_node = candidate_node_chain[0]
    # get all the edges starging from the node
    # all_edges = []
    # edge_names = []
    nodes_to_visit = [start_node]
    while len(nodes_to_visit)>0:
        current_node = nodes_to_visit.pop()
        for successor in graph.successors(current_node):
            # all_edges.append((current_node, successor))
            edge_data = graph.get_edge_data(current_node, successor)
            if edge_data['type'] != 'fixed':
                edge_names.append(edge_data['name'])
            nodes_to_visit.append(successor)
    return list(reversed(edge_names))


def get_urdf_bounding_box(urdf: yourdfpy.URDF):
    """return urdf bounding box
    Example:
        urdf_path=....
        urdf = yourdfpy.URDF.load(urdf_path)
        bounding_box = get_urdf_bounding_box(urdf)
    """
    scene = urdf.scene
    bounding_box = scene.bounding_box
    return bounding_box


def vf_to_mesh(vertices, faces, return_type="o3d"):
    """Converts vertex and face arrays into a mesh object.

    Args:
        vertices: NumPy array (n, 3) of vertex coordinates.
        faces: NumPy array (m, 3) of vertex indices forming triangles.
        return_type: The type of mesh object to return ("o3d" for Open3D, "trimesh" for Trimesh). Defaults to "o3d".

    Returns:
        The mesh object of the specified type:
            - open3d.geometry.TriangleMesh if `return_type` is "o3d".
            - trimesh.Trimesh if `return_type` is "trimesh".

    Raises:
        ValueError: If `return_type` is not "o3d" or "trimesh".
    """
    if return_type == "o3d":
        return o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces),
        )
    elif return_type == "trimesh":
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        raise ValueError("Invalid return_type. Choose 'o3d' or 'trimesh'.")


def triangles_to_vf(triangles):
    """Extracts unique vertices and faces from a NumPy array of triangles.

    Args:
        triangles: NumPy array (n, 3, 3) of triangles, where each triangle is 
                   defined by 3 vertices with (x, y, z) coordinates.

    Returns:
        Tuple (vertices, faces):
            - vertices: NumPy array (num_unique_vertices, 3) of unique vertex coordinates.
            - faces: NumPy array (n, 3) of vertex indices corresponding to the unique vertices.

    Example:
        ```python
        import yourdfpy

        urdf = yourdfpy.URDF.load(...)  
        scene = get_urdf_scene(urdf) 
        vertices, faces = triangles_to_vf(scene.triangles)  
        mesh = vf_to_mesh(vertices, faces, return_type="trimesh")
        mesh.show()
        ```
    """

    # Flatten the triangles array and find unique vertices
    vertices, inverse_indices = np.unique(
        triangles.reshape(-1, 3), axis=0, return_inverse=True
    )

    # Reshape the inverse indices to get the face indices for the unique vertices
    faces = inverse_indices.reshape(-1, 3)

    return vertices, faces


def scene_to_vf(scene: trimesh.Scene):
    """Extracts and transforms vertices and faces from a trimesh Scene.

    Args:
        scene: The trimesh.Scene object to process.

    Returns:
        Tuple (vertices, faces):
            - vertices: NumPy array (num_vertices, 3) of transformed vertex coordinates.
            - faces: NumPy array (num_faces, 3) of vertex indices.

    Example:
        ```python
        import yourdfpy

        urdf = yourdfpy.URDF.load(...)  
        scene = get_urdf_scene(urdf) 
        vertices, faces = scene_to_vf(scene)  
        mesh = vf_to_mesh(vertices, faces, return_type="o3d")
        o3d.visualization.draw_geometries([mesh])  # Visualize using Open3D
        ```
    """
    vertices_list = []
    faces_list = []
    offset = 0

    for node_name in scene.graph.nodes_geometry:
        # Get transform and geometry for the current node
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]

        # Skip if geometry doesn't have triangles
        if not hasattr(geometry, "triangles"):
            continue

        # Apply transform to vertices and append to list
        transformed_vertices = trimesh.transformations.transform_points(
            geometry.vertices.copy(), matrix=transform
        )
        vertices_list.append(transformed_vertices)

        # Append face indices with offset and update offset
        faces_list.append(geometry.faces + offset)
        offset += geometry.vertices.shape[0]

    # Stack vertices and faces into single arrays
    vertices = np.vstack(vertices_list)
    faces = np.vstack(faces_list)
    return vertices, faces


def write_urdf(urdf, new_path, old_path, copy_mesh=False):
    """Writes a URDF file, optionally copying mesh files and adjusting paths.

    Args:
        urdf: the URDF object to write.
        new_path: The path to write the new URDF file to.
        old_path: The path of the original URDF file.
        copy_mesh: If True, mesh files are copied and paths adjusted in the new URDF.
    """
    # --- Path Handling ---
    new_dir, new_urdf_name = os.path.split(os.path.abspath(new_path))
    old_dir, _ = os.path.split(os.path.abspath(
        old_path))  # Old URDF name not used here
    rel_dir = os.path.relpath(old_dir, new_dir)

    os.makedirs(new_dir, exist_ok=True)  # Create directory if it doesn't exist

    # --- Mesh Copying (if enabled) ---
    if copy_mesh:
        new_urdf = copy.copy(urdf)
        new_mesh_dir_rel = "meshes"
        new_mesh_dir = os.path.join(new_dir, new_mesh_dir_rel)
        os.makedirs(new_mesh_dir, exist_ok=True)

        mesh_set = set()
        for link in new_urdf.link_map.values():
            for v in chain(link.collisions, link.visuals):
                if v.geometry and v.geometry.mesh.filename not in mesh_set:
                    # Copy the mesh file and update the path in the URDF
                    old_mesh_path = os.path.join(
                        old_dir, v.geometry.mesh.filename)
                    new_mesh_path = os.path.join(
                        new_mesh_dir, os.path.basename(v.geometry.mesh.filename))
                    shutil.copy2(old_mesh_path, new_mesh_path)
                    v.geometry.mesh.filename = os.path.join(
                        new_mesh_dir_rel, os.path.basename(v.geometry.mesh.filename))
                    mesh_set.add(v.geometry.mesh.filename)

        # Set filename handler to null to avoid errors
        prev_filename_handler = new_urdf._filename_handler
        new_urdf._filename_handler = yourdfpy.filename_handler_null

    # --- URDF Writing ---
    prev_filename_handler = urdf._filename_handler
    if copy_mesh:
        new_urdf.write_xml_file(new_path)
        new_urdf._filename_handler = prev_filename_handler
    else:
        urdf._filename_handler = partial(
            yourdfpy.filename_handler_relative, dir=rel_dir)
        urdf.write_xml_file(new_path)
        urdf._filename_handler = prev_filename_handler
