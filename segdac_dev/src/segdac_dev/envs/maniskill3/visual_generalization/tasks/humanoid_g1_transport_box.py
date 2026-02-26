import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils.building import ground
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.humanoid.transport_box import TransportBoxEnv


@register_env("UnitreeG1TransportBoxTest-v1", max_episode_steps=100)
class TransportBoxTestEnv(TransportBoxEnv):
    def __init__(self, *args, **kwargs):
        self.test_config = kwargs.pop("test_config")
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.ground = ground.build_ground(self.scene, mipmap_levels=7)
        ground_material_config = self.test_config["ground"]["material"]
        ground_texture = ground_material_config["texture"]
        if ground_texture is not None:
            ground_texture = sapien.render.RenderTexture2D(ground_texture)
        ground_base_color = ground_material_config["base_color"]
        if ground_base_color is not None or ground_texture is not None:
            for obj in self.ground._objs:
                render_body_component = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if ground_base_color is not None:
                            part.material.set_base_color(ground_base_color)
                            part.material.set_base_color_texture(None)
                        if ground_texture is not None:
                            part.material.set_base_color_texture(ground_texture)
                        part.material.set_normal_texture(None)
                        part.material.set_emission_texture(None)
                        part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(None)
                        part.material.set_roughness_texture(None)
        # build two tables

        table_model_file = 'assets/table/table.glb'
        scale = 1.2
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.66, 0])
        self.table_1 = builder.build_static(name="table-1")
        table_material_config = self.test_config["table"]["material"]
        table_texture = table_material_config["texture"]
        if table_texture is not None:
            table_texture = sapien.render.RenderTexture2D(table_texture)
        table_base_color = table_material_config["base_color"]
        if table_base_color is not None or table_texture is not None:
            for obj in self.table_1._objs:
                render_body_component = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if table_base_color is not None:
                            part.material.set_base_color(table_base_color)
                            part.material.set_base_color_texture(None)
                        if table_texture is not None:
                            part.material.set_base_color_texture(table_texture)
                        part.material.set_normal_texture(None)
                        part.material.set_emission_texture(None)
                        part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(None)


        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.66, 0])
        self.table_2 = builder.build_static(name="table-2")
        if table_base_color is not None or table_texture is not None:
            for obj in self.table_2._objs:
                render_body_component = obj.find_component_by_type(sapien.render.RenderBodyComponent)
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if table_base_color is not None:
                            part.material.set_base_color(table_base_color)
                            part.material.set_base_color_texture(None)
                        if table_texture is not None:
                            part.material.set_base_color_texture(table_texture)
                        part.material.set_normal_texture(None)
                        part.material.set_emission_texture(None)
                        part.material.set_transmission_texture(None)
                        part.material.set_metallic_texture(None)

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=(0.18, 0.12, 0.12), density=200)
        visual_file = "assets/cardboard_box/box.glb"
        mo_material_config = self.test_config["mo"]["material"]
        mo_base_color = mo_material_config["base_color"]
        mo_texture = mo_material_config["texture"]
        if mo_base_color is None and mo_texture is None:
            mo_material = sapien.render.RenderMaterial()
            mo_material.base_color_texture = sapien.render.RenderTexture2D("assets/cardboard_box/default.png")
        else:
            if mo_base_color is not None:
                mo_material = sapien.render.RenderMaterial(
                    base_color=mo_base_color,
                )
            else:
                mo_material = sapien.render.RenderMaterial()
                mo_material.base_color_texture = sapien.render.RenderTexture2D(mo_texture)
        builder.add_visual_from_file(
            filename=visual_file,
            scale=[0.12] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
            material=mo_material
        )
        builder.initial_pose = sapien.Pose(p=[-0.1, -0.37, 0.7508])
        self.box = builder.build(name="box")
