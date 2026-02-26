from typing import Optional, Union

import sapien
import sapien.render
import numpy as np

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from mani_skill.utils.building.actors.common import _build_by_type


def build_cube(
    scene: ManiSkillScene,
    half_size: float,
    material: sapien.render.RenderMaterial,
    name: str,
    visual_file_path: Optional[str] = None,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[half_size] * 3,
        )
    if visual_file_path is not None:
        builder.add_visual_from_file(
            filename=visual_file_path,
            material=material, 
            scale=[1.0, 1.0, 1.0]
        )
    else:
        builder.add_box_visual(
            half_size=[half_size] * 3,
            material=material,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_red_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    red_color: np.ndarray = np.array([194, 19, 22, 255]) / 255,
    white_color : np.ndarray = np.array([255, 255, 255, 255]) / 255,
    material: Optional[sapien.render.RenderMaterial] = None,
    visual_file_path: Optional[str] = None,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    
    if material is not None and visual_file_path is not None:
        builder.add_visual_from_file(
            filename=visual_file_path,
            material=material, 
            scale=[1.0, 1.0, 1.0]
        )
        if add_collision:
            builder.add_cylinder_collision(
                radius=radius,
                half_length=thickness / 2,
            )
    else:
        builder.add_cylinder_visual(
            radius=radius,
            half_length=thickness / 2,
            material=sapien.render.RenderMaterial(base_color=red_color),
        )
        builder.add_cylinder_visual(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
            material=sapien.render.RenderMaterial(base_color=white_color),
        )
        builder.add_cylinder_visual(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
            material=sapien.render.RenderMaterial(base_color=red_color),
        )
        builder.add_cylinder_visual(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
            material=sapien.render.RenderMaterial(base_color=white_color),
        )
        builder.add_cylinder_visual(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
            material=sapien.render.RenderMaterial(base_color=red_color),
        )
        if add_collision:
            builder.add_cylinder_collision(
                radius=radius,
                half_length=thickness / 2,
            )
            builder.add_cylinder_collision(
                radius=radius * 4 / 5,
                half_length=thickness / 2 + 1e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 3 / 5,
                half_length=thickness / 2 + 2e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 2 / 5,
                half_length=thickness / 2 + 3e-5,
            )
            builder.add_cylinder_collision(
                radius=radius * 1 / 5,
                half_length=thickness / 2 + 4e-5,
            )

    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_two_color_peg(
    scene: ManiSkillScene,
    length: float,
    width: float,
    full_material: sapien.render.RenderMaterial,
    part_1_material: sapien.render.RenderMaterial,
    part_2_material: sapien.render.RenderMaterial,
    name: str,
    visual_file_path: Optional[str],
    body_type: str,
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[length, width, width],
        )
    
    if visual_file_path is not None and full_material is not None:
        builder.add_visual_from_file(
            filename=visual_file_path,
            material=full_material, 
            scale=[1.0, 1.0, 1.0]
        )
    else:
        builder.add_box_visual(
            pose=sapien.Pose(p=[-length / 2, 0, 0]),
            half_size=[length / 2, width, width],
            material=part_1_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[length / 2, 0, 0]),
            half_size=[length / 2, width, width],
            material=part_2_material,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)