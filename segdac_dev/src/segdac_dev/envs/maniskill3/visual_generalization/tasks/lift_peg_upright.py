import sapien
import numpy as np

from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from segdac_dev.envs.maniskill3.visual_generalization.tasks.scene_building import build_two_color_peg


@register_env("LiftPegUprightTest-v1", max_episode_steps=50)
class LiftPegUprightTestEnv(LiftPegUprightEnv):
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02,**kwargs):
        self.test_config = kwargs.pop("test_config")
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        table_material_config = self.test_config["table"]["material"]
        table_texture = table_material_config["texture"]
        if table_texture is not None:
            table_texture = sapien.render.RenderTexture2D(table_texture)
        table_base_color = table_material_config["base_color"]
        if table_base_color is not None or table_texture is not None:
            for obj in self.table_scene.table._objs:
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
                        part.material.set_roughness_texture(None)

        ground_material_config = self.test_config["ground"]["material"]
        ground_texture = ground_material_config["texture"]
        if ground_texture is not None:
            ground_texture = sapien.render.RenderTexture2D(ground_texture)
        ground_base_color = ground_material_config["base_color"]
        if ground_base_color is not None or ground_texture is not None:
            for obj in self.table_scene.ground._objs:
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

        mo_material_config = self.test_config["mo"]["material"]
        mo_texture = mo_material_config["texture"]
        full_material = None
        if mo_texture is not None:
            full_material = sapien.render.RenderMaterial()
            full_material.base_color_texture = sapien.render.RenderTexture2D(mo_texture)
        mo_visual_file_path = mo_material_config["visual_file_path"]
        mo_part_1_material_config = self.test_config["mo"]["part_1"]["material"]
        mo_part_1_base_color = mo_part_1_material_config["base_color"]
        if mo_part_1_base_color is None:
            mo_part_1_base_color = np.array([176, 14, 14, 255]) / 255
        mo_part_1_material = sapien.render.RenderMaterial(
            base_color=mo_part_1_base_color,
        )

        mo_part_2_material_config = self.test_config["mo"]["part_2"]["material"]
        mo_part_2_base_color = mo_part_2_material_config["base_color"]
        if mo_part_2_base_color is None:
            mo_part_2_base_color = np.array([12, 42, 160, 255]) / 255
        mo_part_2_material = sapien.render.RenderMaterial(
            base_color=mo_part_2_base_color,
        )
        self.peg = build_two_color_peg(
            scene=self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            full_material=full_material,
            part_1_material=mo_part_1_material,
            part_2_material=mo_part_2_material,
            name="peg",
            visual_file_path=mo_visual_file_path,
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _load_lighting(self, options: dict):
        ambient_light_color = self.test_config["lighting"]["ambient_light_color"]
        if ambient_light_color is None:
            ambient_light_color = [0.3, 0.3, 0.3]
        self.scene.set_ambient_light(ambient_light_color)

        direction_light_1 = self.test_config["lighting"]["direction"]["light_1"]
        if direction_light_1 is None:
            direction_light_1 = [1, 1, -1]
        direction_light_2 = self.test_config["lighting"]["direction"]["light_2"]
        if direction_light_2 is None:
            direction_light_2 = [0, 0, -1]
        self.scene.add_directional_light(
            direction=direction_light_1,
            color=[1, 1, 1], 
            shadow=self.enable_shadow,
            position=[0, 0, 0], 
            shadow_scale=5, 
            shadow_map_size=2048
        )
        self.scene.add_directional_light(
            direction=direction_light_2,
            color=[1, 1, 1],
            shadow=False,
            position=[0, 0, 0]
        )
